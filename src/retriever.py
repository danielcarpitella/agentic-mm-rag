"""Retriever: text query -> CLIP embedding -> top-k in the FAISS index.

Returns Hits with file paths, not embeddings: the LMM re-encodes the images
with its own vision encoder (see ARCHITECTURE.md §2.2).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# faiss-cpu and torch each link their own copy of libomp: on macOS, this causes
# the process to abort ("OMP: Error #15") or segfault during CLIP's forward pass.
# The two variables must be set BEFORE importing faiss and torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import torch
# CLIPModel is the actual network that turns text and images into comparable embeddings.
# CLIPProcessor prepares inputs before passing them to the model (resizes, normalizes, tokenizes, etc.).
from transformers import CLIPModel, CLIPProcessor 

from .config import RetrieverConfig


@dataclass
class Hit:
    id: str
    image_path: Path
    caption: str
    score: float


class Retriever:
    def __init__(self, cfg: RetrieverConfig):
        if cfg.mode != "text2image": # means: text query → CLIP text embedding → search image embeddings
            raise NotImplementedError(
                f"Mode '{cfg.mode}' is not implemented in the prototype (only 'text2image')."
            )
        index_dir = Path(cfg.index_dir)
        self.index = faiss.read_index(str(index_dir / "images.faiss")) # load the previously created FAISS index
        self.items = json.loads((index_dir / "items.json").read_text()) 
        self.model = CLIPModel.from_pretrained(cfg.encoder) # actual network that turns text and images into comparable embeddings
        self.processor = CLIPProcessor.from_pretrained(cfg.encoder) # prepares inputs for the model (resizes, normalizes, tokenizes, etc.)

    def search(self, query: str, k: int) -> list[Hit]:
        # Turn the query into a CLIP embedding, search FAISS for the k most
        # similar image vectors, and return their data as Hits.


        # return_tensors="pt": asks the processor to return PyTorch tensors, the format CLIPModel expects.
        # padding=True: pads shorter sequences so multiple texts can be processed together at the same length (making the code compatible with multiple queries).
        # truncation=True: if the text is too long, cuts it to the maximum length accepted by CLIP.
        inputs = self.processor(text=[query], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad(): # disable gradient computation because CLIP is not being trained

            # Create embeddings from the processor-prepared inputs; .pooler_output selects CLIP's final text embedding.
            features = self.model.get_text_features(**inputs).pooler_output 

        # Normalize the text embedding so FAISS can compare it with image embeddings using cosine similarity.
        features = features / features.norm(dim=-1, keepdim=True) 

        # scores: similarity score; positions: locations of the most similar embeddings in the index
        scores, positions = self.index.search(features.numpy().astype("float32"), k)

        hits = [] # list for storing search results
        for score, position in zip(scores[0], positions[0]):
            item = self.items[position]
            hits.append(
                Hit(
                    id=item["id"],
                    image_path=Path(item["image"]),
                    caption=item["caption"],
                    score=float(score),
                )
            )
        return hits
