"""Multimodal model wrapper (frozen, inference through prompting only).

The only file that knows which backend runs underneath: MLX on Apple Silicon
or Transformers with CUDA on Windows. The rest of the project stays unaware
of this choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import ModelConfig


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    text: str
    images: list[str] = field(default_factory=list)


class LMM:
    """Frozen multimodal model: no training, only generate()."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

        if cfg.backend == "mlx":
            from mlx_vlm import load
            from mlx_vlm import generate as mlx_generate
            from mlx_vlm.prompt_utils import apply_chat_template

            self._mlx_generate = mlx_generate
            self._mlx_apply_chat_template = apply_chat_template
            self.model, self.processor = load(cfg.name)
        elif cfg.backend == "transformers-cuda":
            self._load_transformers_cuda()
        else:
            raise NotImplementedError(
                f"Backend '{cfg.backend}' is not implemented "
                "(available: 'mlx', 'transformers-cuda')."
            )

    def generate(
        self,
        messages: list[Message],
        max_new_tokens: int | None = None,
    ) -> str:
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
        if self.cfg.backend == "transformers-cuda":
            return self._generate_transformers_cuda(messages, max_new_tokens)

        chat = [{"role": m.role, "content": m.text} for m in messages]
        image_paths = [img for m in messages for img in m.images]
        token_limit = (
            self.cfg.max_new_tokens if max_new_tokens is None else max_new_tokens
        )

        # apply_chat_template converts structured conversation messages into a
        # string in the exact format expected by the Qwen2-VL model.
        formatted_prompt = self._mlx_apply_chat_template(
            self.processor,
            self.model.config,
            chat,
            num_images=len(image_paths),
        )
        result = self._mlx_generate(
            self.model,
            self.processor,
            formatted_prompt,
            image=image_paths or None,
            max_tokens=token_limit,
            temperature=self.cfg.temperature,
            verbose=False,
        )
        return result.text

    def _load_transformers_cuda(self) -> None:
        """Load the native Hugging Face checkpoint in 4-bit on an NVIDIA GPU."""
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError(
                "The 'transformers-cuda' backend requires an NVIDIA GPU with CUDA available."
            )

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self._torch = torch
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.cfg.name,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.cfg.name)

    def _generate_transformers_cuda(
        self,
        messages: list[Message],
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate using the Qwen2-VL Transformers format."""
        from PIL import Image

        chat = []
        images = []
        for message in messages:
            content = [
                {"type": "image", "image": image_path}
                for image_path in message.images
            ]
            content.append({"type": "text", "text": message.text})
            chat.append({"role": message.role, "content": content})

            for image_path in message.images:
                with Image.open(Path(image_path)) as image:
                    images.append(image.convert("RGB"))

        formatted_prompt = self.processor.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[formatted_prompt],
            images=images or None,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generation_args = {
            "max_new_tokens": (
                self.cfg.max_new_tokens
                if max_new_tokens is None
                else max_new_tokens
            ),
            "do_sample": self.cfg.temperature > 0,
        }
        if self.cfg.temperature > 0:
            generation_args["temperature"] = self.cfg.temperature

        with self._torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_args)

        new_tokens = generated_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
