"""Load config.yaml and expose it as typed dataclasses.

Keep it simple: no validation beyond YAML parsing; the file is small and
handwritten by us.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    name: str
    backend: str
    max_new_tokens: int
    temperature: float


@dataclass
class RetrieverConfig:
    encoder: str
    index_backend: str
    index_dir: str
    top_k: int
    mode: str


@dataclass
class OrchestratorConfig:
    max_steps: int
    max_images_in_context: int


@dataclass
class DataConfig:
    images_dir: str
    metadata: str

@dataclass
class Config:
    model: ModelConfig
    retriever: RetrieverConfig
    orchestrator: OrchestratorConfig
    data: DataConfig

def load_config(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        model=ModelConfig(**raw["model"]),
        retriever=RetrieverConfig(**raw["retriever"]),
        orchestrator=OrchestratorConfig(**raw["orchestrator"]),
        data=DataConfig(**raw["data"]),
    )
