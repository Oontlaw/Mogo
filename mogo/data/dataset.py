from pathlib import Path

import numpy as np
import torch


class TokenMemmap:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file() or self.path.stat().st_size % np.dtype(np.uint16).itemsize:
            raise ValueError(f"Missing or invalid uint16 token file: {self.path}")
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")

    def close(self) -> None:
        self.tokens._mmap.close()

    def __enter__(self) -> "TokenMemmap":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.tokens)

    def get_batch(
        self,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.tokens) - sequence_length - 1
        if max_start < 0:
            raise ValueError(
                f"{self.path} has {len(self.tokens)} tokens; "
                f"need at least {sequence_length + 1}"
            )
        starts = rng.integers(0, max_start + 1, size=batch_size)
        samples = np.stack(
            [self.tokens[start : start + sequence_length + 1] for start in starts]
        ).astype(np.int64)
        batch = torch.from_numpy(samples)
        inputs = batch[:, :-1].to(device=device)
        targets = batch[:, 1:].to(device=device)
        return inputs, targets
