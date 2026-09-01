# Agentic Multimodal RAG — prototype

Initial prototype of an Agentic Multimodal RAG system. A small multimodal
model decides when to request visual evidence, retrieves images with
CLIP + FAISS, and uses them to produce the final answer.

> **Current compatibility:** use the MLX backend on Macs with Apple Silicon,
> or the Transformers/CUDA backend on Windows PCs with an NVIDIA GPU. The
> Windows path still needs to be verified on the group's machines.

## Read this first

To understand how the project works and the role of each component, read
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). It contains the complete flow from
the question to image retrieval and the final answer, with diagrams and
examples.

Approved plans for structural project changes are kept in
[`.cursor/plans/`](.cursor/plans/).

## Requirements

- Python 3.12
- Internet connection to install dependencies and download the models
- A few GB of free disk space
- Mac with Apple Silicon, or Windows with an NVIDIA CUDA GPU

On first use, the Qwen2-VL and CLIP weights are downloaded automatically
(about 2–3 GB in total). No API keys are needed.

## Installation: Mac Apple Silicon

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

## Installation: Windows + NVIDIA GPU

Open PowerShell in the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-windows.txt
```

Use an up-to-date NVIDIA driver. The 4-bit model configuration supports the
RTX 4060 (8 GB) and RTX 5070 Ti Laptop GPU (12 GB) in the group.

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

On Windows, select the CUDA configuration:

```powershell
python main.py --config configs/windows-cuda.yaml "What does the Colosseum in Rome look like today?"
```

Each run shows the loop steps in the terminal and saves the complete log
in `logs/`.

## Optional LMM-only test

```bash
python scripts/test_lmm.py
```

On Windows:

```powershell
python scripts/test_lmm.py --config configs/windows-cuda.yaml
```

## Main structure

- `main.py`: entry point
- `src/lmm.py`: multimodal model through MLX or Transformers/CUDA
- `src/retriever.py`: retrieval with CLIP and FAISS
- `src/orchestrator.py`: agentic loop
- `src/prompts.py`: prompts and `SEARCH(...)` protocol
- `scripts/build_index.py`: index construction
- `data/`: mini-dataset images and captions
- `docs/ARCHITECTURE.md`: architecture explanation
- `docs/DIRECTIONS.md`: possible future directions

## Current limitations

- Transformers/CUDA path still awaits validation on the group's Windows PCs;
- designed to run from the project root;
- demonstration dataset limited to landmarks;
- study prototype, not a final product.
