"""Phase 2 test script: a few example queries against the index.

Usage: python scripts/test_retriever.py   (run from the project root)
Checks the Phase 2 definition of done: k visually relevant Hits with correct
paths and captions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.retriever import Retriever

QUERIES = [
    "an ancient Roman amphitheatre",
    "a white marble mausoleum with a dome in India",
    "a modern skyscraper at night",
    "a stone circle in a green field",
]


def main() -> None:
    cfg = load_config("config.yaml")
    retriever = Retriever(cfg.retriever)

    for query in QUERIES:
        print(f"\nQUERY: {query!r}")
        for hit in retriever.search(query, k=cfg.retriever.top_k):
            exists = "ok" if hit.image_path.exists() else "MISSING"
            print(f"  [{hit.score:.3f}] {hit.id} ({hit.image_path}, file {exists})")
            print(f"          {hit.caption[:90]}...")


if __name__ == "__main__":
    main()
