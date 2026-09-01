"""Phase 1 test script: one local image + "describe this image."

Usage: python scripts/test_lmm.py [--config CONFIG]   (run from the project root)
Checks the Phase 1 definition of done: a sensible description in < ~30s.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.lmm import LMM, Message

TEST_IMAGE = Path(__file__).parent / "test_assets" / "test_bicycle.jpg"


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=project_root / "config.yaml", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)

    print(f"Loading model {cfg.model.name} (backend={cfg.model.backend})...")
    t0 = time.time()
    lmm = LMM(cfg.model)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    messages = [Message(role="user", text="Describe this image.", images=[str(TEST_IMAGE)])]

    t0 = time.time()
    output = lmm.generate(messages)
    elapsed = time.time() - t0

    print("\n--- Model response ---")
    print(output)
    print(f"\nGeneration time: {elapsed:.1f}s (Phase 1 criterion: < ~30s)")


if __name__ == "__main__":
    main()
