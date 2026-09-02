from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.config import OrchestratorConfig
from src.lmm import Message
from src.orchestrator import Orchestrator, is_ready
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


class OrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hit = Hit(
            id="sydney_opera_house",
            image_path=Path("data/images/sydney_opera_house.jpg"),
            caption="Sydney Opera House",
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
        image_messages = [message for message in final_messages if message.images]
        self.assertEqual(len(image_messages), 1)
        self.assertNotIn("Image 2:", "\n".join(message.text for message in final_messages))
        self.assertIn(
            "already available as Image 1",
            "\n".join(message.text for message in final_messages),
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
        self.assertIn("FINAL ANSWER - CORRECTION UNCHANGED", log_text)
        self.assertIn("FINAL ANSWER - VALIDATION STILL FAILED", log_text)

    def test_ready_allows_terminal_punctuation_and_trailing_echo(self) -> None:
        self.assertTrue(is_ready("READY."))
        self.assertTrue(is_ready("Ready!\nextra echoed text"))
        self.assertFalse(is_ready("The evidence is READY."))


if __name__ == "__main__":
    unittest.main()
