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
    ANSWER_INSTRUCTION,
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


def extract_search(text: str) -> str | None:
    # Find the first valid SEARCH trigger in the text and return its query.
    for pattern in SEARCH_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


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
        lines.append(ANSWER_INSTRUCTION.format(question=question))
        return Message(
            role="user",
            text="\n".join(lines),
            images=[str(hit.image_path) for hit in hits],
        )

    def run(self, question: str) -> str:
        # Run the loop: generate, interpret SEARCH, retrieve images, and answer.
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
            if query is None:
                # No trigger: either this is the final answer, or the model tried
                # to search with the wrong format (one recovery attempt only).
                if "SEARCH" in output.upper() and not retried:
                    retried = True  # tracks whether a correction attempt has already been made
                    self._log(f"STEP {step} - MALFORMED TRIGGER", "requesting rephrasing")
                    messages.append(Message(role="assistant", text=output))
                    messages.append(Message(role="user", text=RETRY_INSTRUCTION))
                    continue # restart the loop and call the model again
                self._log(f"STEP {step} - FINAL ANSWER", output)
                return output

            self._log(f"STEP {step} - EXTRACTED TRIGGER", query)

            budget = self.cfg.max_images_in_context - images_used

            hits = self.retriever.search(query, k=min(self.top_k, budget))
            self._log(
                f"STEP {step} - RETRIEVED HITS",
                "\n".join(f"[{h.score:.3f}] {h.id} -> {h.image_path}" for h in hits),
            )

            # Save the model's previous response, namely the SEARCH(...) trigger.
            messages.append(Message(role="assistant", text=output))

            # Add a new user message containing retrieved images, captions,
            # labels, and instructions for answering the question.
            messages.append(self._results_message(hits, images_used + 1, question))
            images_used += len(hits)

            if images_used >= self.cfg.max_images_in_context:
                # In this case, on the next loop iteration the model will directly
                # return the output with the final answer.

                messages.append(Message(role="user", text=FINAL_ANSWER_INSTRUCTION))

        # Iteration limit reached: force a final answer.
        messages.append(Message(role="user", text=FINAL_ANSWER_INSTRUCTION))
        final = self.lmm.generate(messages)
        self._log("FINAL ANSWER (forced by step limit)", final)
        return final
