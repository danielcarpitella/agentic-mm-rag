"""Compute CLIP image embeddings and save the FAISS index to disk.

Usage: python scripts/build_index.py   (run from the project root)
Produces: index/images.faiss + index/items.json (records in index order)

CLIP embeddings are used ONLY for searching: the LMM never sees them; it
receives the actual image files (see ARCHITECTURE.md §2.2).
"""

import json
import os
import sys
from pathlib import Path

# faiss-cpu and torch each link their own copy of libomp: on macOS, this causes
# the process to abort ("OMP: Error #15") or segfault during CLIP's forward pass.
# The two variables must be set BEFORE importing faiss and torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    records = [json.loads(line) for line in (ROOT / cfg.data.metadata).open()]
    print(f"{len(records)} items to index")

    model = CLIPModel.from_pretrained(cfg.retriever.encoder)
    processor = CLIPProcessor.from_pretrained(cfg.retriever.encoder)

    images = [Image.open(ROOT / r["image"]).convert("RGB") for r in records]
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        # In transformers 5.x, get_image_features returns an output object:
        # the projected embedding is in pooler_output.
        features = model.get_image_features(**inputs).pooler_output
    # Normalization allows the inner product to be used as cosine similarity.
    features = features / features.norm(dim=-1, keepdim=True)
    embeddings = features.numpy().astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_dir = ROOT / cfg.retriever.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "images.faiss"))
    (index_dir / "items.json").write_text(json.dumps(records, indent=2))

    print(f"Index with {index.ntotal} vectors (dim {embeddings.shape[1]}) saved to {index_dir}")


if __name__ == "__main__":
    main()
