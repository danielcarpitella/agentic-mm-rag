# Agentic Multimodal RAG — prototype

Initial prototype of an Agentic Multimodal RAG system. A small multimodal
model decides when to request visual evidence, retrieves images with
CLIP + FAISS, and uses them to produce the final answer.

> **Current compatibility:** this version was developed and tested
> exclusively on Macs with Apple Silicon (M1/M2/M3/M4). Windows,
> Linux, or Intel Mac users can study the code, but running the entire system
> will require an LMM backend other than MLX.

## Read this first

To understand how the project works and the role of each component, read
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). It contains the complete flow from
the question to image retrieval and the final answer, with diagrams and
examples.

## Requirements

- Mac with Apple Silicon
- Python 3.12
- Internet connection to install dependencies and download the models
- A few GB of free disk space

On first use, the Qwen2-VL and CLIP weights are downloaded automatically
(about 2–3 GB in total). No API keys are needed.

## Installation

Open the terminal in the project root, that is, in the folder containing
this README:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3.12` is not installed and you use Homebrew:

```bash
brew install python@3.12
```

## Index preparation

The landmark mini-dataset is already included in `data/`. The FAISS index is generated
locally and is not included in the repository:

```bash
python scripts/build_index.py
```

To verify the retriever:

```bash
python scripts/test_retriever.py
```

## Running

Run the command from the project root:

```bash
python main.py "What does the Colosseum in Rome look like today?"
```

Each run shows the loop steps in the terminal and saves the complete log
in `logs/`.

## Optional LMM-only test

```bash
python scripts/test_lmm.py
```

## Main structure

- `main.py`: entry point
- `src/lmm.py`: multimodal model through MLX
- `src/retriever.py`: retrieval with CLIP and FAISS
- `src/orchestrator.py`: agentic loop
- `src/prompts.py`: prompts and `SEARCH(...)` protocol
- `scripts/build_index.py`: index construction
- `data/`: mini-dataset images and captions
- `docs/ARCHITECTURE.md`: architecture explanation
- `docs/DIRECTIONS.md`: possible future directions

## Current limitations

- LMM backend available only through MLX;
- designed to run from the project root;
- demonstration dataset limited to landmarks;
- study prototype, not a final product.
