import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class GPTConfig:
    vocab_size: int
    hidden_size: int = 640
    num_layers: int = 10
    num_heads: int = 10
    intermediate_size: int = 1728
    max_sequence_length: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10000.0
    gradient_checkpointing: bool = True


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(norm + self.eps).to(dtype=x.dtype)) * self.weight


def _apply_rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos().to(dtype=x.dtype)[None, None, :, :]
    sin = angles.sin().to(dtype=x.dtype)[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


def _build_rope_cache(
    sequence_length: int, head_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def _apply_rope_cached(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    cos = cos.to(dtype=x.dtype)[None, None, : x.shape[-2], :]
    sin = sin.to(dtype=x.dtype)[None, None, : x.shape[-2], :]
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class SelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.hidden_size % config.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.rope_theta = config.rope_theta
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.dropout

    def forward(
        self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor
    ) -> torch.Tensor:
        batch, sequence, width = x.shape
        q, k, v = self.qkv(x).view(
            batch, sequence, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q = _apply_rope_cached(q, rope_cos, rope_sin)
        k = _apply_rope_cached(k, rope_cos, rope_sin)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size)
        self.attention = SelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config)

    def forward(
        self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), rope_cos, rope_sin)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        rope_cos, rope_sin = _build_rope_cache(
            config.max_sequence_length,
            config.hidden_size // config.num_heads,
            config.rope_theta,
            torch.device("cpu"),
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self.final_norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        residual_scale = 1.0 / math.sqrt(2 * config.num_layers)
        with torch.no_grad():
            for block in self.blocks:
                block.attention.output.weight.mul_(residual_scale)
                block.mlp.down.weight.mul_(residual_scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("Input exceeds configured max_sequence_length")
        x = self.token_embedding(input_ids)
        sequence_length = input_ids.shape[1]
        rope_cos = self.rope_cos[:sequence_length]
        rope_sin = self.rope_sin[:sequence_length]
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(
                    block,
                    x,
                    rope_cos,
                    rope_sin,
                    use_reentrant=False,
                )
            else:
                x = block(x, rope_cos, rope_sin)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
