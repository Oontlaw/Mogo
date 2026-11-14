import math


def cosine_learning_rate(
    step: int,
    learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (learning_rate - min_learning_rate)
