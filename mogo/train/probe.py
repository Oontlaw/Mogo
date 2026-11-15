import argparse
import time
from pathlib import Path
from typing import Any

import torch

from mogo.config import load_config
from mogo.model.gpt import GPT, GPTConfig
from mogo.train.trainer import _optimizer_parameter_groups, _precision_dtype


_MICRO_BATCH_SIZES = (1, 2, 4, 8)
_TRAIN_STEPS = 3


def _build_model(config: dict[str, Any], device: torch.device) -> GPT:
    data_config = config["data"]
    model_config = config["model"]
    return GPT(
        GPTConfig(
            vocab_size=int(data_config["vocab_size"]),
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
            num_heads=int(model_config["num_heads"]),
            intermediate_size=int(model_config["intermediate_size"]),
            max_sequence_length=int(model_config["max_sequence_length"]),
            dropout=float(model_config["dropout"]),
            rope_theta=float(model_config["rope_theta"]),
            gradient_checkpointing=bool(model_config["gradient_checkpointing"]),
        )
    ).to(device)


def _train_step(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch_size: int,
    sequence_length: int,
    accumulation: int,
    vocab_size: int,
    amp_dtype: torch.dtype | None,
    gradient_clip: float,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    for _ in range(accumulation):
        tokens = torch.randint(
            vocab_size,
            (batch_size, sequence_length + 1),
            device=next(model.parameters()).device,
        )
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        with torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("Model did not return a training loss")
        scaler.scale(loss / accumulation).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    scaler.step(optimizer)
    scaler.update()


def probe(config_path: str | Path) -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; memory probe requires a CUDA device.")
        return

    config = load_config(config_path)
    training_config = config["training"]
    device = torch.device("cuda")
    amp_dtype = _precision_dtype(training_config["precision"], device)
    model = _build_model(config, device)
    optimizer = torch.optim.AdamW(
        _optimizer_parameter_groups(model, float(training_config["weight_decay"])),
        lr=float(training_config["learning_rate"]),
        betas=(float(training_config["beta1"]), float(training_config["beta2"])),
        foreach=False,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=training_config["precision"] == "fp16"
    )
    model.train()
    sequence_length = int(training_config["sequence_length"])
    accumulation = int(training_config["gradient_accumulation_steps"])
    vocab_size = int(config["data"]["vocab_size"])
    gradient_clip = float(training_config["gradient_clip"])
    largest_working_size = None

    for batch_size in _MICRO_BATCH_SIZES:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_time = time.perf_counter()
        try:
            for _ in range(_TRAIN_STEPS):
                _train_step(
                    model,
                    optimizer,
                    scaler,
                    batch_size,
                    sequence_length,
                    accumulation,
                    vocab_size,
                    amp_dtype,
                    gradient_clip,
                )
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError as error:
            message = str(error).splitlines()[0]
            del error
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"OOM at micro_batch_size={batch_size}: {message}")
            break

        elapsed = time.perf_counter() - start_time
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        tokens_per_second = (
            _TRAIN_STEPS * batch_size * sequence_length * accumulation / elapsed
        )
        print(
            f"micro_batch_size={batch_size}: "
            f"peak_memory={peak_memory_gb:.2f} GB, "
            f"tokens/sec={tokens_per_second:.2f}"
        )
        largest_working_size = batch_size

    result = str(largest_working_size) if largest_working_size is not None else "none"
    print(f"Largest micro_batch_size that worked: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe MOGO CUDA training memory")
    parser.add_argument("--config", default="configs/mogo.yaml")
    args = parser.parse_args()
    probe(args.config)


if __name__ == "__main__":
    main()
