import argparse
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from mogo.config import load_config
from mogo.data.dataset import TokenMemmap
from mogo.eval.generate import generate
from mogo.model.gpt import GPT, GPTConfig, count_parameters
from mogo.train.schedule import cosine_learning_rate


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str, allow_cpu: bool = False) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError(
            "CUDA was not found for device='auto'. "
            f"torch.__version__={torch.__version__}, "
            f"torch.version.cuda={torch.version.cuda}. "
            "Check the driver with `nvidia-smi` and this interpreter with "
            '`python -c "import torch; print(torch.__version__, '
            'torch.version.cuda, torch.cuda.is_available())"`. '
            "Use --allow-cpu only for CPU smoke runs."
        )
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _precision_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    if precision == "fp32":
        return None
    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was selected but this CUDA device does not support it")
        return torch.bfloat16
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 training requires CUDA; use bf16 or fp32 on CPU")
        return torch.float16
    raise ValueError(f"Unsupported precision: {precision}")


def _checkpoint_state(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
) -> dict:
    return {
        "step": step,
        "model_config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _optimizer_parameter_groups(model: GPT, weight_decay: float) -> list[dict]:
    decay_parameters = []
    no_decay_parameters = []
    embedding_weight = model.token_embedding.weight
    seen_parameters = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen_parameters:
            continue
        seen_parameters.add(id(parameter))
        if (
            parameter.ndim == 2
            and name.endswith(".weight")
            and parameter is not embedding_weight
        ):
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)
    print(
        "AdamW parameter tensors: "
        f"decay={len(decay_parameters)}, no_decay={len(no_decay_parameters)}"
    )
    return [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]


def _save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


@torch.no_grad()
def _evaluate(
    model: GPT,
    data: TokenMemmap,
    batches: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    seed: int,
    amp_dtype: torch.dtype | None,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed + 2)
    model.eval()
    losses = []
    for _ in range(batches):
        inputs, targets = data.get_batch(batch_size, sequence_length, device, rng)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("Model did not return a validation loss")
        losses.append(float(loss.item()))
    model.train()
    mean_loss = sum(losses) / len(losses)
    return mean_loss, math.exp(mean_loss)


def train(
    config_path: str | Path, resume: bool = False, allow_cpu: bool = False
) -> None:
    config = load_config(config_path)
    _set_seed(int(config["seed"]))
    data_config, model_config, training_config = (
        config["data"],
        config["model"],
        config["training"],
    )
    device = _device(training_config["device"], allow_cpu=allow_cpu)
    amp_dtype = _precision_dtype(training_config["precision"], device)
    data_dir = Path(data_config["output_dir"])
    train_data = TokenMemmap(data_dir / "train.bin")
    val_data = TokenMemmap(data_dir / "val.bin")
    tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()
    sequence_length = int(training_config["sequence_length"])
    if sequence_length > int(model_config["max_sequence_length"]):
        raise ValueError("training.sequence_length exceeds model.max_sequence_length")
    if sequence_length < 1:
        raise ValueError("training.sequence_length must be positive")
    if len(train_data) < sequence_length + 1 or len(val_data) < sequence_length + 1:
        raise ValueError(
            "Both train and validation token files must contain at least "
            f"{sequence_length + 1} tokens"
        )
    for key in ("micro_batch_size", "gradient_accumulation_steps", "max_steps"):
        if int(training_config[key]) < 1:
            raise ValueError(f"training.{key} must be positive")
    for key in ("eval_interval", "eval_batches", "save_interval", "sample_interval"):
        if int(training_config[key]) < 1:
            raise ValueError(f"training.{key} must be positive")

    model = GPT(
        GPTConfig(
            vocab_size=vocab_size,
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
    print(f"Device: {device}; trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        _optimizer_parameter_groups(model, float(training_config["weight_decay"])),
        lr=float(training_config["learning_rate"]),
        betas=(float(training_config["beta1"]), float(training_config["beta2"])),
        foreach=False,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=training_config["precision"] == "fp16"
    )
    checkpoint_dir = Path(training_config["output_dir"])
    last_checkpoint = checkpoint_dir / "last.pt"
    best_checkpoint = checkpoint_dir / "best.pt"
    start_step = 0
    best_val_loss = math.inf
    batch_rng_state = None
    if resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {last_checkpoint}")
        state = torch.load(last_checkpoint, map_location=device, weights_only=False)
        schedule_keys = (
            "max_steps",
            "learning_rate",
            "min_learning_rate",
            "warmup_steps",
        )
        changed_schedule = [
            f"{key}: saved={state.get(key, '<missing>')}, "
            f"current={training_config[key]}"
            for key in schedule_keys
            if state.get(key) != training_config[key]
        ]
        if changed_schedule:
            print(
                "Warning: LR schedule has changed since the checkpoint "
                f"({'; '.join(changed_schedule)})."
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_step = int(state["step"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in state["cuda_rng"]]
            )
        batch_rng_state = state["batch_rng_state"]
        del state
        if best_checkpoint.is_file():
            best_state = torch.load(
                best_checkpoint, map_location="cpu", weights_only=False
            )
            best_val_loss = float(best_state.get("val_loss", math.inf))
            del best_state
        print(f"Resuming from step {start_step}")

    batch_size = int(training_config["micro_batch_size"])
    eval_batch_size = int(training_config.get("eval_batch_size", batch_size))
    if eval_batch_size < 1:
        raise ValueError("training.eval_batch_size must be positive")
    accumulation = int(training_config["gradient_accumulation_steps"])
    max_steps = int(training_config["max_steps"])
    batch_rng = np.random.default_rng(int(config["seed"]) + 1)
    if resume:
        batch_rng.bit_generator.state = batch_rng_state

    training_start_time = time.perf_counter()
    log_interval_start = training_start_time
    tokens_since_log = 0
    peak_gpu_memory_reported = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(start_step, max_steps):
        learning_rate = cosine_learning_rate(
            step,
            float(training_config["learning_rate"]),
            float(training_config["min_learning_rate"]),
            int(training_config["warmup_steps"]),
            max_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(accumulation):
            inputs, targets = train_data.get_batch(
                batch_size, sequence_length, device, batch_rng
            )
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                _, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("Model did not return a training loss")
            total_loss += float(loss.detach().item())
            scaler.scale(loss / accumulation).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training_config["gradient_clip"])
        )
        scaler.step(optimizer)
        scaler.update()

        completed_step = step + 1
        tokens_since_log += batch_size * sequence_length * accumulation
        if completed_step % 10 == 0:
            now = time.perf_counter()
            interval_elapsed = now - log_interval_start
            elapsed = now - training_start_time
            tokens_per_second = tokens_since_log / max(interval_elapsed, 1e-9)
            print(
                f"step {completed_step}/{max_steps} "
                f"loss {total_loss / accumulation:.4f} lr {learning_rate:.6g} "
                f"tokens/sec {tokens_per_second:.2f} elapsed {elapsed:.1f}s"
            )
            tokens_since_log = 0
            log_interval_start = now
        if completed_step % int(training_config["eval_interval"]) == 0:
            val_loss, perplexity = _evaluate(
                model,
                val_data,
                int(training_config["eval_batches"]),
                eval_batch_size,
                sequence_length,
                device,
                int(config["seed"]),
                amp_dtype,
            )
            print(f"validation loss {val_loss:.4f} perplexity {perplexity:.2f}")
            if device.type == "cuda" and not peak_gpu_memory_reported:
                peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")
                peak_gpu_memory_reported = True
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    "step": completed_step,
                    "model_config": asdict(model.config),
                    "model": model.state_dict(),
                    "val_loss": val_loss,
                }
                _save_checkpoint(best_checkpoint, best_state)
                print(f"Saved best checkpoint to {best_checkpoint}")
        if completed_step % int(training_config["sample_interval"]) == 0:
            prompt = str(config["generation"]["prompt"])
            print(
                "sample: "
                + generate(
                    model,
                    tokenizer,
                    prompt,
                    int(config["generation"]["max_new_tokens"]),
                    float(config["generation"]["temperature"]),
                    int(config["generation"]["top_k"]),
                )
            )
        if (
            completed_step % int(training_config["save_interval"]) == 0
            or completed_step == max_steps
        ):
            checkpoint_state = _checkpoint_state(model, optimizer, scaler, completed_step)
            checkpoint_state["batch_rng_state"] = batch_rng.bit_generator.state
            checkpoint_state.update(
                {
                    "max_steps": max_steps,
                    "learning_rate": float(training_config["learning_rate"]),
                    "min_learning_rate": float(training_config["min_learning_rate"]),
                    "warmup_steps": int(training_config["warmup_steps"]),
                }
            )
            _save_checkpoint(last_checkpoint, checkpoint_state)
            print(f"Saved checkpoint to {last_checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MOGO")
    parser.add_argument("--config", default="configs/mogo.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    train(args.config, resume=args.resume, allow_cpu=args.allow_cpu)


if __name__ == "__main__":
    main()
