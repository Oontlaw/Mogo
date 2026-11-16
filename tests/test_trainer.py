import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from mogo.data.dataset import TokenMemmap
from mogo.model.gpt import GPT, GPTConfig
from mogo.data.prepare import prepare
from mogo.eval.generate import main as generate_main
from mogo.train.trainer import _device, _evaluate, train


class TrainerTests(unittest.TestCase):
    def test_auto_device_requires_cuda_unless_cpu_is_allowed(self):
        with patch("mogo.train.trainer.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA was not found") as error:
                _device("auto")
            self.assertIn(str(torch.__version__), str(error.exception))
            self.assertIn(str(torch.version.cuda), str(error.exception))
            self.assertIn("nvidia-smi", str(error.exception))
            self.assertEqual(_device("auto", allow_cpu=True), torch.device("cpu"))

    def test_validation_batches_repeat_without_changing_training_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "val.bin"
            np.arange(64, dtype=np.uint16).tofile(path)
            model = GPT(
                GPTConfig(
                    vocab_size=64,
                    hidden_size=16,
                    num_layers=1,
                    num_heads=2,
                    intermediate_size=32,
                    max_sequence_length=4,
                    gradient_checkpointing=False,
                )
            )
            training_rng = np.random.default_rng(11)
            training_rng_state = training_rng.bit_generator.state.copy()
            with TokenMemmap(path) as data:
                first = _evaluate(
                    model,
                    data,
                    batches=2,
                    batch_size=3,
                    sequence_length=4,
                    device=torch.device("cpu"),
                    seed=17,
                    amp_dtype=None,
                )
                second = _evaluate(
                    model,
                    data,
                    batches=2,
                    batch_size=3,
                    sequence_length=4,
                    device=torch.device("cpu"),
                    seed=17,
                    amp_dtype=None,
                )
            self.assertEqual(first, second)
            self.assertEqual(training_rng.bit_generator.state, training_rng_state)

    def test_resume_matches_uninterrupted_training_and_cli_loads_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "books"
            data_dir = root / "processed"
            input_dir.mkdir()
            for index in range(3):
                (input_dir / f"book-{index}.txt").write_text(
                    (f"Distinct book {index} text for deterministic training. " * 20),
                    encoding="utf-8",
                )
            prep_config_path = root / "prepare.yaml"
            prep_config_path.write_text(
                yaml.safe_dump(
                    {
                        "seed": 23,
                        "data": {
                            "input_dir": str(input_dir),
                            "output_dir": str(data_dir),
                            "validation_fraction": 0.34,
                            "vocab_size": 300,
                            "min_token_frequency": 2,
                            "max_split_rate": 10000.0,
                            "max_single_rate": 10000.0,
                            "min_words": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            prepare(prep_config_path)
            model_config = {
                "hidden_size": 8,
                "num_layers": 1,
                "num_heads": 2,
                "intermediate_size": 16,
                "max_sequence_length": 4,
                "dropout": 0.0,
                "rope_theta": 10000.0,
                "gradient_checkpointing": False,
            }

            def write_training_config(output_dir: Path, max_steps: int) -> Path:
                config_path = root / f"{output_dir.name}.yaml"
                config_path.write_text(
                    yaml.safe_dump(
                        {
                            "seed": 23,
                            "data": {"output_dir": str(data_dir)},
                            "model": model_config,
                            "training": {
                                "output_dir": str(output_dir),
                                "device": "cpu",
                                "precision": "fp32",
                                "micro_batch_size": 1,
                                "gradient_accumulation_steps": 1,
                                "sequence_length": 4,
                                "max_steps": max_steps,
                                "learning_rate": 0.001,
                                "min_learning_rate": 0.001,
                                "warmup_steps": 0,
                                "weight_decay": 0.1,
                                "beta1": 0.9,
                                "beta2": 0.95,
                                "gradient_clip": 1.0,
                                "eval_interval": 2,
                                "eval_batches": 1,
                                "save_interval": 2,
                                "sample_interval": 100,
                            },
                            "generation": {
                                "prompt": "Book",
                                "max_new_tokens": 1,
                                "temperature": 0.8,
                                "top_k": 5,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return config_path

            straight_dir = root / "straight"
            resumed_dir = root / "resumed"
            straight_config = write_training_config(straight_dir, 10)
            resumed_config = write_training_config(resumed_dir, 2)
            straight_output = io.StringIO()
            with contextlib.redirect_stdout(straight_output):
                train(straight_config)
            self.assertIn("tokens/sec", straight_output.getvalue())
            self.assertIn("elapsed", straight_output.getvalue())
            train(resumed_config)

            resumed_config = write_training_config(resumed_dir, 10)
            resume_output = io.StringIO()
            with contextlib.redirect_stdout(resume_output):
                train(resumed_config, resume=True)
            self.assertIn("Warning: LR schedule has changed", resume_output.getvalue())
            self.assertIn("max_steps: saved=2, current=10", resume_output.getvalue())

            straight_state = torch.load(
                straight_dir / "last.pt", map_location="cpu", weights_only=False
            )
            resumed_state = torch.load(
                resumed_dir / "last.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(straight_state["step"], 10)
            self.assertEqual(resumed_state["step"], 10)
            self.assertEqual(resumed_state["max_steps"], 10)
            self.assertEqual(resumed_state["learning_rate"], 0.001)
            self.assertEqual(resumed_state["min_learning_rate"], 0.001)
            self.assertEqual(resumed_state["warmup_steps"], 0)
            self.assertEqual(straight_state["model_config"], resumed_state["model_config"])
            for name, straight_weight in straight_state["model"].items():
                self.assertTrue(
                    torch.equal(straight_weight, resumed_state["model"][name]), name
                )

            model = GPT(GPTConfig(**straight_state["model_config"]))
            model.load_state_dict(straight_state["model"])
            with TokenMemmap(data_dir / "val.bin") as validation_data:
                straight_loss = _evaluate(
                    model,
                    validation_data,
                    batches=1,
                    batch_size=1,
                    sequence_length=4,
                    device=torch.device("cpu"),
                    seed=23,
                    amp_dtype=None,
                )[0]
            resumed_model = GPT(GPTConfig(**resumed_state["model_config"]))
            resumed_model.load_state_dict(resumed_state["model"])
            with TokenMemmap(data_dir / "val.bin") as validation_data:
                resumed_loss = _evaluate(
                    resumed_model,
                    validation_data,
                    batches=1,
                    batch_size=1,
                    sequence_length=4,
                    device=torch.device("cpu"),
                    seed=23,
                    amp_dtype=None,
                )[0]
            self.assertEqual(straight_loss, resumed_loss)

            best_state = torch.load(
                straight_dir / "best.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(best_state["model_config"], straight_state["model_config"])
            self.assertEqual(
                set(best_state), {"step", "model_config", "model", "val_loss"}
            )
            for checkpoint in ("best.pt", "last.pt"):
                output = io.StringIO()
                with patch.object(
                    sys,
                    "argv",
                    [
                        "mogo.eval.generate",
                        "--checkpoint",
                        str(straight_dir / checkpoint),
                        "--tokenizer",
                        str(data_dir / "tokenizer.json"),
                        "--prompt",
                        "Book",
                        "--max-new-tokens",
                        "1",
                    ],
                ), contextlib.redirect_stdout(output):
                    generate_main()
                self.assertIn("Book", output.getvalue())


if __name__ == "__main__":
    unittest.main()
