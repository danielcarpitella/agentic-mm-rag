"""Multimodal model wrapper (frozen, inference through prompting only).

The only file that knows which backend runs underneath: for now, only MLX
(Apple Silicon). A future "transformers-mps" backend should be added here,
selected by config.yaml, without touching the rest of the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mlx_vlm import generate as mlx_generate
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template

from .config import ModelConfig


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    text: str
    images: list[str] = field(default_factory=list)


class LMM:
    """Frozen multimodal model: no training, only generate()."""

    def __init__(self, cfg: ModelConfig):
        if cfg.backend != "mlx":
            raise NotImplementedError(
                f"Backend '{cfg.backend}' is not implemented in the prototype (only 'mlx')."
            )
        self.cfg = cfg
        self.model, self.processor = load(cfg.name)

    def generate(self, messages: list[Message]) -> str:
        """messages may contain text and images (local paths).

        Roles are passed to the model's chat template (the standard format the
        model uses to distinguish conversation roles such as system, user, and
        assistant). Concatenating everything into a single turn caused the 2B
        model to ignore the system prompt (for example, it was not told which
        content was from the system and which was from the user, so it did not
        understand that the initial text contained instructions on how to answer).
        Images are placed in the final user turn, where the orchestrator inserts
        the retrieval results.
        """
        chat = [{"role": m.role, "content": m.text} for m in messages]
        image_paths = [img for m in messages for img in m.images]

        # apply_chat_template converts structured conversation messages into a
        # string in the exact format expected by the Qwen2-VL model.
        formatted_prompt = apply_chat_template(
            self.processor,
            self.model.config,
            chat,
            num_images=len(image_paths),
        )
        result = mlx_generate(
            self.model,
            self.processor,
            formatted_prompt,
            image=image_paths or None,
            max_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            verbose=False,
        )
        return result.text
