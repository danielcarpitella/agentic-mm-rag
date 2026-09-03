from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.config import OrchestratorConfig
from src.lmm import Message
from src.orchestrator import Orchestrator, has_valid_image_citations, is_ready
from src.retriever import Hit


class FakeLMM:
    def __init__(self, outputs: list[str]):
        self.outputs = iter(outputs)
        self.calls: list[tuple[int | None, list[Message]]] = []

    def generate(
        self,
        messages: list[Message],
        max_new_tokens: int | None = None,
    ) -> str:
        snapshot = [
            Message(role=message.role, text=message.text, images=list(message.images))
            for message in messages
        ]
        self.calls.append((max_new_tokens, snapshot))
        return next(self.outputs)


class FakeRetriever:
    def __init__(self, hit: Hit):
        self.hit = hit
        self.queries: list[str] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.queries.append(query)
        return [self.hit][:k]


class MappingRetriever:
    def __init__(self, hits_by_query: dict[str, Hit]):
        self.hits_by_query = hits_by_query
        self.queries: list[str] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.queries.append(query)
        return [self.hits_by_query[query]][:k]


class OrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hit = Hit(
            id="sydney_opera_house",
            image_path=Path("data/images/sydney_opera_house.jpg"),
            caption="UNIQUE CAPTION TEXT THAT MUST NOT REACH THE FINAL CONTEXT",
            score=0.9,
        )

    def test_duplicate_id_is_not_reinserted_and_model_decides_again(self) -> None:
        lmm = FakeLMM(
            [
                'SEARCH("Sydney Opera House roof")',
                'SEARCH("roof of the Sydney Opera House")',
                "READY.",
                "The roof has curved sail-like shells (Image 1).",
            ]
        )
        retriever = FakeRetriever(self.hit)
        cfg = OrchestratorConfig(
            max_steps=4,
            max_images_in_context=4,
            decision_max_new_tokens=48,
        )

        with TemporaryDirectory() as log_dir:
            with redirect_stdout(StringIO()):
                answer = Orchestrator(
                    lmm, retriever, cfg, top_k=1, log_dir=log_dir
                ).run("Describe the roof.")

        self.assertEqual(
            retriever.queries,
            ["Sydney Opera House roof", "roof of the Sydney Opera House"],
        )
        self.assertEqual(answer, "The roof has curved sail-like shells (Image 1).")
        self.assertEqual([call[0] for call in lmm.calls], [48, 48, 48, None])

        final_messages = lmm.calls[-1][1]
        self.assertEqual([message.role for message in final_messages], ["system", "user"])
        image_messages = [message for message in final_messages if message.images]
        self.assertEqual(len(image_messages), 1)
        self.assertEqual(
            image_messages[0].images,
            ["data/images/sydney_opera_house.jpg"],
        )
        final_text = "\n".join(message.text for message in final_messages)
        self.assertIn("Image 1 — sydney opera house", final_text)
        self.assertIn('begin exactly with "(Image 1):"', final_text)
        self.assertNotIn("UNIQUE CAPTION TEXT", final_text)
        self.assertNotIn("already available as Image 1", final_text)
        self.assertNotIn('SEARCH("', final_text)

    def test_on_event_hook_reports_the_loop_sequence_without_changing_it(self) -> None:
        lmm = FakeLMM(
            [
                'SEARCH("Sydney Opera House roof")',
                'SEARCH("roof of the Sydney Opera House")',
                "READY.",
                "The roof has curved sail-like shells (Image 1).",
            ]
        )
        retriever = FakeRetriever(self.hit)
        cfg = OrchestratorConfig(
            max_steps=4,
            max_images_in_context=4,
            decision_max_new_tokens=48,
        )
        events: list[dict] = []

        with TemporaryDirectory() as log_dir:
            with redirect_stdout(StringIO()):
                answer = Orchestrator(
                    lmm, retriever, cfg, top_k=1, log_dir=log_dir, on_event=events.append
                ).run("Describe the roof.")

        # Same behaviour as without the hook.
        self.assertEqual(answer, "The roof has curved sail-like shells (Image 1).")
        self.assertEqual(len(retriever.queries), 2)

        self.assertEqual(
            [event["type"] for event in events],
            [
                "question",
                "decision",
                "search",
                "retrieval",
                "decision",
                "search",
                "duplicate",
                "decision",
                "ready",
                "final_prompt",
                "answer",
            ],
        )
        retrieval = events[3]
        self.assertEqual(retrieval["step"], 0)
        self.assertEqual(
            retrieval["hits"],
            [
                {
                    "label": 1,
                    "id": "sydney_opera_house",
                    "score": 0.9,
                    "image_path": "data/images/sydney_opera_house.jpg",
                    "caption": self.hit.caption,
                }
            ],
        )
        self.assertEqual(events[6], {"type": "duplicate", "step": 1, "labels": [1], "new_evidence": False})
        self.assertEqual(events[9], {"type": "final_prompt", "labels": [1]})
        self.assertEqual(
            events[10],
            {
                "type": "answer",
                "text": answer,
                "valid": True,
                "corrected": False,
                "no_evidence": False,
            },
        )

    def test_duplicate_attempts_consume_steps_and_reach_step_limit(self) -> None:
        lmm = FakeLMM(
            [
                'SEARCH("first wording")',
                'SEARCH("second wording")',
                'SEARCH("third wording")',
                "The visible roof is curved (Image 1).",
            ]
        )
        retriever = FakeRetriever(self.hit)
        cfg = OrchestratorConfig(
            max_steps=3,
            max_images_in_context=4,
            decision_max_new_tokens=48,
        )

        with TemporaryDirectory() as log_dir:
            with redirect_stdout(StringIO()):
                answer = Orchestrator(
                    lmm, retriever, cfg, top_k=1, log_dir=log_dir
                ).run("Describe the roof.")

        self.assertEqual(len(retriever.queries), 3)
        self.assertEqual(answer, "The visible roof is curved (Image 1).")
        self.assertEqual([call[0] for call in lmm.calls], [48, 48, 48, None])

    def test_invalid_citation_gets_only_one_correction(self) -> None:
        unchanged_answer = "The roof is curved."
        lmm = FakeLMM(
            [
                'SEARCH("Sydney Opera House")',
                "READY",
                unchanged_answer,
                unchanged_answer,
            ]
        )
        retriever = FakeRetriever(self.hit)
        cfg = OrchestratorConfig(
            max_steps=3,
            max_images_in_context=4,
            decision_max_new_tokens=48,
        )

        with TemporaryDirectory() as log_dir:
            orchestrator = Orchestrator(lmm, retriever, cfg, top_k=1, log_dir=log_dir)
            with redirect_stdout(StringIO()):
                answer = orchestrator.run("Describe the roof.")
            log_text = orchestrator.log_path.read_text()

        self.assertEqual(answer, unchanged_answer)
        self.assertEqual(len(lmm.calls), 4)
        initial_final_messages = lmm.calls[-2][1]
        correction_messages = lmm.calls[-1][1]
        self.assertEqual(len(initial_final_messages), 2)
        self.assertEqual(len(correction_messages), 2)
        correction_text = "\n".join(message.text for message in correction_messages)
        self.assertNotIn(unchanged_answer, correction_text)
        self.assertNotIn("UNIQUE CAPTION TEXT", correction_text)
        self.assertEqual(
            correction_messages[-1].images,
            ["data/images/sydney_opera_house.jpg"],
        )
        self.assertIn("FINAL ANSWER - CORRECTION UNCHANGED", log_text)
        self.assertIn("FINAL ANSWER - VALIDATION STILL FAILED", log_text)

    def test_failed_regeneration_does_not_replace_original_answer(self) -> None:
        original = "The roof has several visible curved sections."
        invalid_regeneration = "A different answer still without any citation."
        lmm = FakeLMM(
            [
                'SEARCH("Sydney Opera House")',
                "READY",
                original,
                invalid_regeneration,
            ]
        )
        cfg = OrchestratorConfig(
            max_steps=3,
            max_images_in_context=4,
            decision_max_new_tokens=48,
        )

        with TemporaryDirectory() as log_dir:
            orchestrator = Orchestrator(
                lmm,
                FakeRetriever(self.hit),
                cfg,
                top_k=1,
                log_dir=log_dir,
            )
            with redirect_stdout(StringIO()):
                answer = orchestrator.run("Describe the roof.")
            log_text = orchestrator.log_path.read_text()

        self.assertEqual(answer, original)
        self.assertEqual(len(lmm.calls), 4)
        self.assertIn("FINAL ANSWER - KEEPING ORIGINAL", log_text)

    def test_final_context_preserves_two_image_label_order(self) -> None:
        colosseum = Hit(
            id="colosseum",
            image_path=Path("data/images/colosseum.jpg"),
            caption="FIRST SECRET CAPTION",
            score=0.9,
        )
        sydney = Hit(
            id="sydney_opera_house",
            image_path=Path("data/images/sydney_opera_house.jpg"),
            caption="SECOND SECRET CAPTION",
            score=0.8,
        )
        lmm = FakeLMM(
            [
                'SEARCH("Colosseum")',
                'SEARCH("Sydney Opera House")',
                "READY",
                (
                    "The first facade has stacked arches (Image 1). "
                    "The second has white shell-like roof forms (Image 2)."
                ),
            ]
        )
        retriever = MappingRetriever(
            {"Colosseum": colosseum, "Sydney Opera House": sydney}
        )
        cfg = OrchestratorConfig(
            max_steps=4,
            max_images_in_context=4,
            decision_max_new_tokens=48,
        )

        with TemporaryDirectory() as log_dir:
            with redirect_stdout(StringIO()):
                answer = Orchestrator(
                    lmm, retriever, cfg, top_k=1, log_dir=log_dir
                ).run("Compare the two landmarks.")

        self.assertTrue(has_valid_image_citations(answer, 2))
        final_messages = lmm.calls[-1][1]
        self.assertEqual([message.role for message in final_messages], ["system", "user"])
        self.assertEqual(
            final_messages[-1].images,
            [
                "data/images/colosseum.jpg",
                "data/images/sydney_opera_house.jpg",
            ],
        )
        final_text = "\n".join(message.text for message in final_messages)
        self.assertLess(
            final_text.index("Image 1 — colosseum"),
            final_text.index("Image 2 — sydney opera house"),
        )
        self.assertIn('begin exactly with "(Image 1):"', final_text)
        self.assertIn('begin exactly with "(Image 2):"', final_text)
        self.assertNotIn("FIRST SECRET CAPTION", final_text)
        self.assertNotIn("SECOND SECRET CAPTION", final_text)
        self.assertNotIn("SEARCH", final_text)

    def test_final_answer_validation_requires_complete_parenthetical_labels(self) -> None:
        self.assertTrue(
            has_valid_image_citations(
                "One facade has arches (Image 1). Another has shells (Image 2).",
                2,
            )
        )
        self.assertFalse(
            has_valid_image_citations("Only one facade has arches (Image 1).", 2)
        )
        self.assertFalse(
            has_valid_image_citations(
                "One facade has arches (Image 1). Another is tall (Image 3).",
                2,
            )
        )
        self.assertFalse(
            has_valid_image_citations(
                "Image 1 shows a facade with several stacked arches.", 1
            )
        )
        self.assertFalse(
            has_valid_image_citations(
                "A visible description is followed by Image N.", 1
            )
        )
        self.assertFalse(has_valid_image_citations("(Image 1).", 1))
        self.assertFalse(
            has_valid_image_citations(
                "Required citation tokens are repeated here (Image 1).", 1
            )
        )

    def test_ready_allows_terminal_punctuation_and_trailing_echo(self) -> None:
        self.assertTrue(is_ready("READY."))
        self.assertTrue(is_ready("Ready!\nextra echoed text"))
        self.assertFalse(is_ready("The evidence is READY."))


if __name__ == "__main__":
    unittest.main()
