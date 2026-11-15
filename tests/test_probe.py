import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from mogo.train.probe import main


class ProbeTests(unittest.TestCase):
    def test_cpu_path_reports_cuda_unavailable(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "mogo.yaml"
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["mogo.train.probe", "--config", str(config_path)]),
            patch("mogo.train.probe.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(output),
        ):
            main()
        self.assertIn("CUDA not available", output.getvalue())


if __name__ == "__main__":
    unittest.main()
