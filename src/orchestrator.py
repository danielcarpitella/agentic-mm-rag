"""The agentic loop: the heart of the project (see ARCHITECTURE.md §2.3).

It depends only on the LMM and Retriever interfaces: changing the model, runtime,
or index type should not require changes here.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .config import OrchestratorConfig
from .lmm import LMM, Message
from .prompts import (
    CORRECT_ANSWER_INSTRUCTION,
    DECISION_INSTRUCTION,
    FINAL_ANSWER_INSTRUCTION,
    RESULTS_HEADER,
    RETRY_INSTRUCTION,
    SYSTEM_PROMPT,
)
from .retriever import Hit, Retriever

# Try the canonical format first, followed by increasingly permissive forms:
# with a 2B model, the syntax may vary even when the request is understandable.
# Examples that can all be interpreted as the query "the Colosseum in Rome":
#   SEARCH("the Colosseum in Rome")  -> expected format
#   SEARCH('the Colosseum in Rome')  -> single quotes
#   SEARCH(the Colosseum in Rome)    -> no quotes
#   SEARCH: the Colosseum in Rome    -> colon instead of parentheses
# Order matters: the precise format is selected before the generic one, so the
# query is extracted in the least ambiguous way possible.
SEARCH_PATTERNS = [
    re.compile(r'SEARCH\s*\(\s*"([^"]+)"\s*\)'),
    re.compile(r"SEARCH\s*\(\s*'([^']+)'\s*\)"),
    re.compile(r"SEARCH\s*\(\s*([^)]+?)\s*\)"),
    re.compile(r"SEARCH\s*[:\-]\s*(.+)"),
]
IMAGE_LABEL_PATTERN = re.compile(r"\bImage\s+(\d+)\b", re.IGNORECASE)


def extract_search(text: str) -> str | None:
    # Find the first valid SEARCH trigger in the text and return its query.
    for pattern in SEARCH_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def is_ready(text: str) -> bool:
    """READY is valid only when it is the complete decision."""
    return text.strip().upper() == "READY"


def has_valid_image_citations(text: str, images_used: int) -> bool:
    """Require at least one citation and reject labels not in the context."""
    labels = [int(label) for label in IMAGE_LABEL_PATTERN.findall(text)]
    return bool(labels) and all(1 <= label <= images_used for label in labels)


class Orchestrator:
    def __init__(
        self,
        lmm: LMM,
        retriever: Retriever,
        cfg: OrchestratorConfig,
        top_k: int,
        log_dir: str | Path = "logs",
    ):
        # Connect the LMM and retriever and prepare the session log file.
        self.lmm = lmm
        self.retriever = retriever
        self.cfg = cfg
        self.top_k = top_k
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

    def _log(self, section: str, body: str) -> None:
        """Log to stdout and a file to show the LITERAL string that reaches the
        model; otherwise, debugging the prompt is impossible."""
        # Write the same event to both the terminal and the persistent log.
        text = f"\n===== {section} =====\n{body}"
        print(text)
        with self.log_path.open("a") as f:
            f.write(text + "\n")

    def _results_message(self, hits: list[Hit], first_label: int, question: str) -> Message:
        # Turn the retrieved hits (class Hit) into the next step's multimodal
        # message (with image paths, descriptions, instructions, etc.).
        lines = [RESULTS_HEADER]
        for offset, hit in enumerate(hits):
            lines.append(f"Image {first_label + offset}: {hit.caption}")
        lines.append(DECISION_INSTRUCTION.format(question=question))
        return Message(
            role="user",
            text="\n".join(lines),
            images=[str(hit.image_path) for hit in hits],
        )

    @staticmethod
    def _available_labels(images_used: int) -> str:
        return ", ".join(f"Image {label}" for label in range(1, images_used + 1))

    def _generate_final(
        self,
        messages: list[Message],
        question: str,
        images_used: int,
    ) -> str:
        available_labels = self._available_labels(images_used)
        instruction = FINAL_ANSWER_INSTRUCTION.format(
            question=question,
            available_labels=available_labels or "none",
        )
        messages.append(Message(role="user", text=instruction))
        self._log("FINAL ANSWER - PROMPT SENT", instruction)

        output = self.lmm.generate(messages)
        self._log("FINAL ANSWER - RAW OUTPUT", output)
        if has_valid_image_citations(output, images_used):
            self._log("FINAL ANSWER", output)
            return output

        self._log(
            "FINAL ANSWER - INVALID CITATIONS",
            f"available labels: {available_labels or 'none'}",
        )
        messages.append(Message(role="assistant", text=output))
        correction = CORRECT_ANSWER_INSTRUCTION.format(
            available_labels=available_labels or "none"
        )
        messages.append(Message(role="user", text=correction))
        corrected = self.lmm.generate(messages)
        self._log("FINAL ANSWER - CORRECTED OUTPUT", corrected)

        if not has_valid_image_citations(corrected, images_used):
            self._log("FINAL ANSWER - VALIDATION STILL FAILED", corrected)
        return corrected

    def run(self, question: str) -> str:
        # Alternate one model decision with at most one retrieval. Answering is
        # a separate generation step reached only through READY or a hard limit.
        messages = [
            Message(role="system", text=SYSTEM_PROMPT),
            Message(role="user", text=question),
        ]
        images_used = 0
        retried = False

        self._log("QUESTION", question)

        for step in range(self.cfg.max_steps):
            self._log(f"STEP {step} - PROMPT SENT", "\n".join(m.text for m in messages))
            output = self.lmm.generate(messages)
            self._log(f"STEP {step} - RAW OUTPUT", output)

            query = extract_search(output)
            if query is None and is_ready(output):
                self._log(f"STEP {step} - READY", "model declared evidence sufficient")
                messages.append(Message(role="assistant", text="READY"))
                if images_used:
                    return self._generate_final(messages, question, images_used)

            if query is None:
                if not retried:
                    retried = True
                    self._log(
                        f"STEP {step} - INVALID DECISION",
                        "requesting exactly one SEARCH or READY",
                    )
                    messages.append(Message(role="assistant", text=output))
                    messages.append(Message(role="user", text=RETRY_INSTRUCTION))
                    continue

                self._log(f"STEP {step} - DECISION RETRY FAILED", output)
                if images_used:
                    return self._generate_final(messages, question, images_used)
                return output

            self._log(f"STEP {step} - EXTRACTED TRIGGER", query)

            budget = self.cfg.max_images_in_context - images_used
            if budget <= 0:
                self._log(f"STEP {step} - IMAGE LIMIT", "forcing final answer")
                return self._generate_final(messages, question, images_used)

            hits = self.retriever.search(query, k=min(self.top_k, budget))
            self._log(
                f"STEP {step} - RETRIEVED HITS",
                "\n".join(f"[{h.score:.3f}] {h.id} -> {h.image_path}" for h in hits),
            )

            # Preserve only the parsed action actually executed. A small model can
            # emit extra SEARCH(...) lines, which must not look completed in the
            # next turn's conversation history.
            messages.append(Message(role="assistant", text=f'SEARCH("{query}")'))

            # Add a new user message containing retrieved images, captions,
            # labels, and instructions for answering the question.
            messages.append(self._results_message(hits, images_used + 1, question))
            images_used += len(hits)

            if images_used >= self.cfg.max_images_in_context:
                self._log(f"STEP {step} - IMAGE LIMIT", "forcing final answer")
                return self._generate_final(messages, question, images_used)

        self._log("STEP LIMIT", "forcing final answer")
        return self._generate_final(messages, question, images_used)
