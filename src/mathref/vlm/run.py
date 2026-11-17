"""Run a frozen VLM over a prepared input manifest (the external model-call step).

`prepare_vlm_inputs` builds answer-free rows (labeled image + prompt); this calls
a model on each and writes raw responses that `parse_vlm_outputs` turns into
predictions. Backends are pluggable:

- MockVLMBackend: no API/GPU, returns a deterministic ranking so the whole chain
  (prepare -> run -> parse -> evaluate) can be tested today.
- OpenAIVLMBackend: calls the OpenAI vision API (needs `OPENAI_API_KEY`).
- QwenVLMBackend: self-hosted Qwen2.5-VL via Hugging Face Transformers.
- SarashinaVLMBackend: self-hosted sarashina2-vision (Japanese VLM) via
  Hugging Face Transformers (`trust_remote_code`).

The choice of model is recorded so the eval `meta.model` can cite it.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Protocol

from mathref.utils.io import read_jsonl, write_jsonl


class VLMBackend(Protocol):
    name: str
    model: str

    def load(self) -> None:
        """Initialize the client/model once before the first call (no-op if N/A)."""

    def generate(self, row: dict[str, Any]) -> str:
        """Return the raw model response for one manifest row."""


class MockVLMBackend:
    """No-API stand-in: deterministically ranks the candidate labels in reverse."""

    name = "mock"
    model = "mock"

    def load(self) -> None:  # pragma: no cover - trivial
        return None

    def generate(self, row: dict[str, Any]) -> str:
        labels = [candidate["label"] for candidate in row["candidate_labels"]]
        return json.dumps({"ranking": list(reversed(labels))})


class OpenAIVLMBackend:
    """Calls the OpenAI vision chat API. Requires the `openai` package and a key."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self._client: Any = None
        self.last_usage: dict[str, Any] | None = None

    def load(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            raise RuntimeError("openai is not installed. Run `pip install openai`.") from exc
        self._client = OpenAI()  # reads OPENAI_API_KEY from the environment

    def generate(self, row: dict[str, Any]) -> str:
        image_bytes = Path(row["labeled_image_path"]).read_bytes()
        data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": row["prompt"]},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        usage = getattr(response, "usage", None)
        self.last_usage = None
        if usage is not None:
            # served_model is the dated snapshot actually used; record it so the
            # eval meta and the paper can pin the exact version.
            self.last_usage = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "served_model": getattr(response, "model", self.model),
            }
        return response.choices[0].message.content or ""


class QwenVLMBackend:
    """Self-hosted Qwen2.5-VL on a GPU host."""

    name = "qwen"

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._engine: Any = None
        self._processor: Any = None

    def load(self) -> None:
        if self._engine is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - depends on GPU host
            raise RuntimeError(
                "Qwen dependencies are not installed. On the GPU host: "
                "`pip install torch transformers accelerate`."
            ) from exc

        if torch.cuda.is_available():
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            dtype = torch.float32
        self._engine = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model,
            torch_dtype=dtype,
            device_map=self.device or "auto",
            attn_implementation="sdpa",
        )
        self._engine.eval()
        self._processor = AutoProcessor.from_pretrained(self.model)

    def generate(self, row: dict[str, Any]) -> str:
        if self._engine is None or self._processor is None:
            raise RuntimeError("Qwen backend is not loaded. Call load() before generate().")
        image_path = Path(row["labeled_image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Labeled image does not exist: {image_path}")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": row["prompt"]},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._engine.device)
        generated_ids = self._engine.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        completion_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        return self._processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


class SarashinaVLMBackend:
    """Self-hosted sarashina2-vision (SB Intuitions' Japanese VLM) on a GPU host.

    Follows the model-card usage flow (verified 2026-07-04 at
    https://huggingface.co/sbintuitions/sarashina2-vision-8b), which differs
    from Qwen's: the chat template renders the text prompt only (the template
    itself inserts the `<|file|>` image slot), the image is passed through the
    processor call, and generation stops at the model's "\\n###" turn delimiter
    via the processor's custom stopping criteria. Both the model and the
    processor ship custom code, hence `trust_remote_code=True`.
    """

    name = "sarashina"

    def __init__(
        self,
        model: str = "sbintuitions/sarashina2-vision-8b",
        device: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._engine: Any = None
        self._processor: Any = None

    def load(self) -> None:
        if self._engine is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:  # pragma: no cover - depends on GPU host
            raise RuntimeError(
                "sarashina dependencies are not installed. On the GPU host: "
                "`pip install torch transformers accelerate` "
                "(the model card pins transformers==4.47.0)."
            ) from exc

        self._processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
        self._engine = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype="auto",
            device_map=self.device or "auto",
            trust_remote_code=True,
        )
        self._engine.eval()

    def generate(self, row: dict[str, Any]) -> str:
        if self._engine is None or self._processor is None:
            raise RuntimeError("sarashina backend is not loaded. Call load() before generate().")
        image_path = Path(row["labeled_image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Labeled image does not exist: {image_path}")
        from PIL import Image  # core dependency; imported here to keep module import light

        messages = [{"role": "user", "content": row["prompt"]}]
        text_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
        inputs = self._processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self._engine.device)
        stopping_criteria = self._processor.get_stopping_criteria(["\n###"])
        generated_ids = self._engine.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            stopping_criteria=stopping_criteria,
        )
        completion_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        return self._processor.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]


BACKENDS = {
    "mock": MockVLMBackend,
    "openai": OpenAIVLMBackend,
    "qwen": QwenVLMBackend,
    "sarashina": SarashinaVLMBackend,
}


def get_backend(name: str, **kwargs: Any) -> VLMBackend:
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose from {sorted(BACKENDS)}.")
    return BACKENDS[name](**kwargs)


def run_vlm(
    inputs_path: str | Path,
    backend: VLMBackend,
    usage_sink: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Call the backend on each manifest row; return raw-output rows.

    Token usage is metering, not part of the raw-output contract
    (vlm_output.schema.json is closed), so it goes to `usage_sink` instead of
    the rows when the backend reports it.
    """
    backend.load()
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(inputs_path):
        rows.append({"example_id": row["example_id"], "response": backend.generate(row)})
        last_usage = getattr(backend, "last_usage", None)
        if usage_sink is not None and last_usage:
            usage_sink.append({"example_id": row["example_id"], **last_usage})
    return rows


def run_vlm_resumable(
    inputs_path: str | Path,
    output_path: str | Path,
    backend: VLMBackend,
    usage_path: str | Path | None = None,
) -> tuple[int, int]:
    """Checkpoint each response and skip completed examples on a resumed run."""
    inputs = list(read_jsonl(inputs_path))
    input_ids = [row["example_id"] for row in inputs]
    if len(input_ids) != len(set(input_ids)):
        raise ValueError("Input manifest contains duplicate example_id values.")

    out = Path(output_path)
    existing_rows = list(read_jsonl(out)) if out.is_file() else []
    completed_ids = [row["example_id"] for row in existing_rows]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("Existing output contains duplicate example_id values.")
    unknown_ids = sorted(set(completed_ids) - set(input_ids))
    if unknown_ids:
        raise ValueError(
            f"Existing output contains {len(unknown_ids)} IDs absent from the input manifest."
        )

    completed_set = set(completed_ids)
    pending = [row for row in inputs if row["example_id"] not in completed_set]
    if not pending:
        return len(completed_ids), len(inputs)

    out.parent.mkdir(parents=True, exist_ok=True)
    usage_out = Path(usage_path) if usage_path is not None else None
    if usage_out is not None:
        usage_out.parent.mkdir(parents=True, exist_ok=True)

    backend.load()
    completed = len(completed_ids)
    with out.open("a", encoding="utf-8") as output_stream:
        usage_stream = (
            usage_out.open("a", encoding="utf-8") if usage_out is not None else None
        )
        try:
            for row in pending:
                result = {
                    "example_id": row["example_id"],
                    "response": backend.generate(row),
                }
                output_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_stream.flush()
                completed += 1

                last_usage = getattr(backend, "last_usage", None)
                if usage_stream is not None and last_usage:
                    usage_stream.write(
                        json.dumps(
                            {"example_id": row["example_id"], **last_usage},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    usage_stream.flush()

                if completed % 25 == 0 or completed == len(inputs):
                    print(f"VLM checkpoint: {completed}/{len(inputs)} responses")
        finally:
            if usage_stream is not None:
                usage_stream.close()
    return completed, len(inputs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a frozen VLM over a prepared input manifest.")
    parser.add_argument("--inputs", required=True, help="VLM input manifest JSONL from prepare_vlm_inputs.")
    parser.add_argument("--out", required=True, help="Raw VLM output JSONL.")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="mock")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Backend model name (e.g. gpt-4o, Qwen/Qwen2.5-VL-7B-Instruct, "
            "sbintuitions/sarashina2-vision-8b)."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default=None, help="Device for self-hosted backends, e.g. cuda:0.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation limit for self-hosted backends.")
    parser.add_argument(
        "--usage-out",
        default=None,
        help="Optional JSONL for per-call token usage (API backends only); kept separate from the raw-output contract.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Checkpoint each output row and skip example IDs already present in --out.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    kwargs: dict[str, Any] = {}
    if args.backend == "openai":
        if args.model:
            kwargs["model"] = args.model
        kwargs["temperature"] = args.temperature
    elif args.backend in ("qwen", "sarashina"):
        if args.model:
            kwargs["model"] = args.model
        kwargs["device"] = args.device
        kwargs["max_new_tokens"] = args.max_new_tokens
    backend = get_backend(args.backend, **kwargs)
    if args.resume:
        completed, total = run_vlm_resumable(
            args.inputs,
            args.out,
            backend,
            usage_path=args.usage_out,
        )
        print(
            f"VLM ({backend.name}:{backend.model}): "
            f"{completed}/{total} responses checkpointed -> {args.out}"
        )
        return

    usage_rows: list[dict[str, Any]] = []
    rows = run_vlm(args.inputs, backend, usage_sink=usage_rows)
    write_jsonl(args.out, rows)
    print(f"VLM ({backend.name}:{backend.model}): wrote {len(rows)} responses -> {args.out}")
    if args.usage_out and usage_rows:
        write_jsonl(args.usage_out, usage_rows)
        total_in = sum(r["prompt_tokens"] for r in usage_rows)
        total_out = sum(r["completion_tokens"] for r in usage_rows)
        print(f"usage: {total_in} prompt + {total_out} completion tokens over {len(usage_rows)} calls -> {args.usage_out}")


if __name__ == "__main__":
    main()
