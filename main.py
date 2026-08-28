"""Prototype CLI: python main.py "question..."."""

import argparse
import os

# Same OpenMP workaround as src/retriever.py: it is needed here because main.py
# is the first file loaded and imports both mlx and torch/faiss.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from src.config import load_config
from src.lmm import LMM
from src.orchestrator import Orchestrator
from src.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser() 
    parser.add_argument("question") 
    args = parser.parse_args() # args contains the command-line arguments, including the user's question

    cfg = load_config("config.yaml")
    orchestrator = Orchestrator(
        lmm=LMM(cfg.model),
        retriever=Retriever(cfg.retriever),
        cfg=cfg.orchestrator,
        top_k=cfg.retriever.top_k,
    )

    answer = orchestrator.run(args.question)
    print("\n===== ANSWER =====")
    print(answer) # prints the model's final answer (the one after all the reasoning)
    print(f"\n(full log in {orchestrator.log_path})")


if __name__ == "__main__":
    main()
