"""Run the four-question minimal end-to-end evaluation.

Usage: python scripts/evaluate_minimal.py [--config CONFIG]

This is a small manual evaluation, not an automated benchmark. For each test,
inspect the SEARCH actions, retrieved IDs, and whether the final answer cites
only images that were actually retrieved.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.lmm import LMM
from src.orchestrator import Orchestrator
from src.retriever import Retriever


QUESTIONS = [
    (
        "single retrieval — roof shape",
        "Describe the visible shape of the Sydney Opera House roof. "
        "Base your answer on retrieved visual evidence and cite the image used.",
        ("sydney_opera_house",),
    ),
    (
        "single retrieval — stone arrangement",
        "Describe the visible arrangement of the stones at Stonehenge. "
        "Base your answer on retrieved visual evidence and cite the image used.",
        ("stonehenge",),
    ),
    (
        "multiple retrievals — ancient and modern architecture",
        "Compare the visible architecture of the Colosseum in Rome and the "
        "Sydney Opera House. Retrieve separate visual evidence for each landmark, "
        "one landmark at a time, and cite both images in the final answer.",
        ("colosseum", "sydney_opera_house"),
    ),
    (
        "multiple retrievals — monumental figures",
        "Compare the visible pose and overall form of the Statue of Liberty and "
        "Christ the Redeemer. Retrieve separate visual evidence for each monument, "
        "one monument at a time, and cite both images in the final answer.",
        ("statue_of_liberty", "christ_the_redeemer_statue"),
    ),
]

SEPARATOR = "=" * 88


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=project_root / "config.yaml", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("Loading the model and retriever once for all four tests...")
    lmm = LMM(cfg.model)
    retriever = Retriever(cfg.retriever)

    for number, (name, question, expected_ids) in enumerate(QUESTIONS, start=1):
        print(f"\n{SEPARATOR}")
        print(f"BEGIN TEST {number}/{len(QUESTIONS)} — {name}")
        print(f"QUESTION: {question}")
        print(f"EXPECTED RETRIEVED IDS: {', '.join(expected_ids)}")
        print(SEPARATOR)

        orchestrator = Orchestrator(
            lmm=lmm,
            retriever=retriever,
            cfg=cfg.orchestrator,
            top_k=cfg.retriever.top_k,
        )
        answer = orchestrator.run(question)

        print(f"\n{SEPARATOR}")
        print(f"END TEST {number}/{len(QUESTIONS)} — {name}")
        print("FINAL ANSWER:")
        print(answer)
        print(f"LOG: {orchestrator.log_path}")
        print(
            "MANUAL CHECK: one SEARCH per decision; expected IDs retrieved; "
            "final answer cites only the available Image labels."
        )
        print(SEPARATOR)


if __name__ == "__main__":
    main()
