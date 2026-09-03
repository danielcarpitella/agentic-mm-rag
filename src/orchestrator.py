"""The agentic loop: the heart of the project (see ARCHITECTURE.md §2.3).

It depends only on the LMM and Retriever interfaces: changing the model, runtime,
or index type should not require changes here.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import OrchestratorConfig
from .lmm import LMM, Message
from .prompts import (
    CORRECT_ANSWER_INSTRUCTION,
    DECISION_INSTRUCTION,
    DUPLICATE_RESULT_INSTRUCTION,
    FINAL_ANSWER_INSTRUCTION,
    FINAL_ANSWER_SYSTEM_PROMPT,
    NO_EVIDENCE_FINAL_ANSWER_INSTRUCTION,
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
PARENTHETICAL_IMAGE_LABEL_PATTERN = re.compile(
    r"\(\s*Image\s+(\d+)\s*\)", re.IGNORECASE
)
IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\bImage\s+N\b", re.IGNORECASE)
INSTRUCTION_ECHO_FRAGMENTS = (
    "required citation tokens",
    "mandatory citation tokens",
    "image identities",
    "write one concise visual observation",
    "sentence 1:",
)


def extract_search(text: str) -> str | None:
    # Find the first valid SEARCH trigger in the text and return its query.
    for pattern in SEARCH_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def is_ready(text: str) -> bool:
    """Accept READY (optionally punctuated) when it is the first decision."""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return re.fullmatch(r"READY[.!]?", first_line, re.IGNORECASE) is not None


def has_valid_image_citations(text: str, images_used: int) -> bool:
    """Validate complete parenthetical citations and reject degenerate output."""
    if images_used <= 0 or IMAGE_PLACEHOLDER_PATTERN.search(text):
        return False

    expected_labels = set(range(1, images_used + 1))
    citation_matches = list(PARENTHETICAL_IMAGE_LABEL_PATTERN.finditer(text))
    citation_labels = {int(match.group(1)) for match in citation_matches}
    if citation_labels != expected_labels:
        return False

    # Every occurrence of "Image N" must be part of a parenthetical citation.
    citation_spans = [match.span() for match in citation_matches]
    for mention in IMAGE_LABEL_PATTERN.finditer(text):
        start, end = mention.span()
        if not any(
            span_start <= start and end <= span_end
            for span_start, span_end in citation_spans
        ):
            return False

    # A citation by itself is not a visual observation.
    content_without_citations = PARENTHETICAL_IMAGE_LABEL_PATTERN.sub("", text)
    words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", content_without_citations)
    if len(words) < 5:
        return False

    lowered = text.lower()
    return not any(fragment in lowered for fragment in INSTRUCTION_ECHO_FRAGMENTS)


class Orchestrator:
    def __init__(
        self,
        lmm: LMM,
        retriever: Retriever,
        cfg: OrchestratorConfig,
        top_k: int,
        log_dir: str | Path = "logs",
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ):
        # Connect the LMM and retriever and prepare the session log file.
        self.lmm = lmm
        self.retriever = retriever
        self.cfg = cfg
        self.top_k = top_k
        # Optional structured-event sink (used by the demo UI for streaming).
        # It mirrors what _log already records and never changes the loop.
        self.on_event = on_event
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

    def _emit(self, event_type: str, **data: Any) -> None:
        """Forward a structured event to the optional sink (no-op by default)."""
        if self.on_event is not None:
            self.on_event({"type": event_type, **data})

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

    @staticmethod
    def _response_structure(evidence: list[tuple[int, Hit]]) -> str:
        """Build label-specific guidance without supplying copyable visual claims."""
        lines = []
        for sentence_number, (label, hit) in enumerate(evidence, start=1):
            landmark = hit.id.replace("_", " ")
            lines.append(
                f"Sentence {sentence_number}: begin exactly with "
                f'"(Image {label}):" and then describe one visible detail of {landmark}.'
            )
        if len(evidence) > 1:
            citations = " and ".join(f"(Image {label})" for label, _ in evidence)
            lines.append(
                f'Final sentence: begin exactly with "{citations}:" and compare only '
                "the visible details already described."
            )
        return "\n".join(lines)

    @staticmethod
    def _image_mapping(evidence: list[tuple[int, Hit]]) -> str:
        return "\n".join(
            f"Image {label} — {hit.id.replace('_', ' ')}" for label, hit in evidence
        )

    @staticmethod
    def _required_citations(evidence: list[tuple[int, Hit]]) -> str:
        return ", ".join(f"(Image {label})" for label, _ in evidence)

    @staticmethod
    def _messages_for_log(messages: list[Message]) -> str:
        blocks = []
        for message in messages:
            block = [f"[{message.role.upper()}]", message.text]
            if message.images:
                block.append("Attached images: " + ", ".join(message.images))
            blocks.append("\n".join(block))
        return "\n\n".join(blocks)

    def _final_messages(
        self,
        question: str,
        evidence: list[tuple[int, Hit]],
        *,
        correction: bool = False,
    ) -> list[Message]:
        template = (
            CORRECT_ANSWER_INSTRUCTION if correction else FINAL_ANSWER_INSTRUCTION
        )
        instruction = template.format(
            question=question,
            image_mapping=self._image_mapping(evidence),
            required_citations=self._required_citations(evidence),
            response_structure=self._response_structure(evidence),
        )
        return [
            Message(role="system", text=FINAL_ANSWER_SYSTEM_PROMPT),
            Message(
                role="user",
                text=instruction,
                images=[str(hit.image_path) for _, hit in evidence],
            ),
        ]

    def _generate_final(
        self,
        decision_messages: list[Message],
        question: str,
        evidence: list[tuple[int, Hit]],
    ) -> str:
        if not evidence:
            fallback_messages = [
                *decision_messages,
                Message(
                    role="user",
                    text=NO_EVIDENCE_FINAL_ANSWER_INSTRUCTION.format(question=question),
                ),
            ]
            self._log(
                "FINAL ANSWER - NO EVIDENCE PROMPT",
                self._messages_for_log(fallback_messages),
            )
            output = self.lmm.generate(fallback_messages)
            self._log("FINAL ANSWER - RAW OUTPUT", output)
            self._emit("answer", text=output, valid=False, corrected=False, no_evidence=True)
            return output

        images_used = len(evidence)
        available_labels = self._available_labels(images_used)
        final_messages = self._final_messages(question, evidence)
        self._log(
            "FINAL ANSWER - PROMPT SENT",
            self._messages_for_log(final_messages),
        )

        self._emit("final_prompt", labels=[label for label, _ in evidence])
        output = self.lmm.generate(final_messages)
        self._log("FINAL ANSWER - RAW OUTPUT", output)
        if has_valid_image_citations(output, images_used):
            self._log("FINAL ANSWER", output)
            self._emit("answer", text=output, valid=True, corrected=False, no_evidence=False)
            return output

        self._log(
            "FINAL ANSWER - INVALID",
            f"required labels: {available_labels}",
        )
        self._emit("invalid_answer", text=output, labels=[label for label, _ in evidence])
        correction_messages = self._final_messages(
            question,
            evidence,
            correction=True,
        )
        self._log(
            "FINAL ANSWER - REGENERATION PROMPT SENT",
            self._messages_for_log(correction_messages),
        )
        corrected = self.lmm.generate(correction_messages)
        self._log("FINAL ANSWER - CORRECTED OUTPUT", corrected)

        if corrected.strip() == output.strip():
            self._log(
                "FINAL ANSWER - CORRECTION UNCHANGED",
                "the bounded retry reproduced the original answer",
            )

        if not has_valid_image_citations(corrected, images_used):
            self._log("FINAL ANSWER - VALIDATION STILL FAILED", corrected)
            self._log(
                "FINAL ANSWER - KEEPING ORIGINAL",
                "the unvalidated regeneration did not replace the first answer",
            )
            self._emit("answer", text=output, valid=False, corrected=False, no_evidence=False)
            return output

        self._log("FINAL ANSWER", corrected)
        self._emit("answer", text=corrected, valid=True, corrected=True, no_evidence=False)
        return corrected

    def run(self, question: str) -> str:
        # Alternate one model decision with at most one retrieval. Answering is
        # a separate generation step reached only through READY or a hard limit.
        messages = [
            Message(role="system", text=SYSTEM_PROMPT),
            Message(role="user", text=question),
        ]
        images_used = 0
        retrieved_evidence: list[tuple[int, Hit]] = []
        seen_image_labels: dict[str, int] = {}
        retried = False

        self._log("QUESTION", question)
        self._emit("question", text=question)

        for step in range(self.cfg.max_steps):
            self._log(f"STEP {step} - PROMPT SENT", "\n".join(m.text for m in messages))
            output = self.lmm.generate(
                messages,
                max_new_tokens=self.cfg.decision_max_new_tokens,
            )
            self._log(f"STEP {step} - RAW OUTPUT", output)
            self._emit("decision", step=step, raw=output)

            query = extract_search(output)
            if query is None and is_ready(output):
                self._log(f"STEP {step} - READY", "model declared evidence sufficient")
                self._emit("ready", step=step)
                messages.append(Message(role="assistant", text="READY"))
                if images_used:
                    return self._generate_final(messages, question, retrieved_evidence)

            if query is None:
                if not retried:
                    retried = True
                    self._log(
                        f"STEP {step} - INVALID DECISION",
                        "requesting exactly one SEARCH or READY",
                    )
                    self._emit("invalid_decision", step=step, raw=output)
                    messages.append(Message(role="assistant", text=output))
                    messages.append(Message(role="user", text=RETRY_INSTRUCTION))
                    continue

                self._log(f"STEP {step} - DECISION RETRY FAILED", output)
                self._emit("decision_failed", step=step, raw=output)
                if images_used:
                    return self._generate_final(messages, question, retrieved_evidence)
                return output

            self._log(f"STEP {step} - EXTRACTED TRIGGER", query)
            self._emit("search", step=step, query=query)

            budget = self.cfg.max_images_in_context - images_used
            if budget <= 0:
                self._log(f"STEP {step} - IMAGE LIMIT", "forcing final answer")
                self._emit("limit", step=step, reason="image_limit")
                return self._generate_final(messages, question, retrieved_evidence)

            hits = self.retriever.search(query, k=min(self.top_k, budget))
            self._log(
                f"STEP {step} - RETRIEVED HITS",
                "\n".join(f"[{h.score:.3f}] {h.id} -> {h.image_path}" for h in hits),
            )

            # Preserve only the parsed action actually executed. A small model can
            # emit extra SEARCH(...) lines, which must not look completed in the
            # next turn's conversation history.
            messages.append(Message(role="assistant", text=f'SEARCH("{query}")'))

            duplicate_labels = []
            new_hits = []
            for hit in hits:
                if hit.id in seen_image_labels:
                    duplicate_labels.append(seen_image_labels[hit.id])
                    continue
                label = images_used + len(new_hits) + 1
                seen_image_labels[hit.id] = label
                new_hits.append(hit)
                retrieved_evidence.append((label, hit))

            if duplicate_labels:
                existing_labels = ", ".join(
                    f"Image {label}" for label in sorted(set(duplicate_labels))
                )
                self._log(
                    f"STEP {step} - DUPLICATE HITS",
                    f"{existing_labels} already in context; no new label assigned",
                )
                self._emit(
                    "duplicate",
                    step=step,
                    labels=sorted(set(duplicate_labels)),
                    new_evidence=bool(new_hits),
                )

            if duplicate_labels and not new_hits:
                feedback = DUPLICATE_RESULT_INSTRUCTION.format(
                    existing_labels=existing_labels
                )
                messages.append(Message(role="user", text=feedback))
                continue

            # Add a new user message containing retrieved images, captions,
            # labels, and instructions for answering the question.
            first_label = images_used + 1
            messages.append(self._results_message(new_hits, first_label, question))
            images_used += len(new_hits)
            self._emit(
                "retrieval",
                step=step,
                hits=[
                    {
                        "label": first_label + offset,
                        "id": hit.id,
                        "score": hit.score,
                        "image_path": str(hit.image_path),
                        "caption": hit.caption,
                    }
                    for offset, hit in enumerate(new_hits)
                ],
            )

            if images_used >= self.cfg.max_images_in_context:
                self._log(f"STEP {step} - IMAGE LIMIT", "forcing final answer")
                self._emit("limit", step=step, reason="image_limit")
                return self._generate_final(messages, question, retrieved_evidence)

        self._log("STEP LIMIT", "forcing final answer")
        self._emit("limit", step=self.cfg.max_steps, reason="step_limit")
        return self._generate_final(messages, question, retrieved_evidence)
