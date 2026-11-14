import torch
from tokenizers import Tokenizer

from mogo.data.tokenizer import END_OF_BOOK
from mogo.model.gpt import GPT


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    device = next(model.parameters()).device
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long, device=device)
    if input_ids.shape[1] == 0:
        raise ValueError("Generation prompt must encode to at least one token")
    was_training = model.training
    end_id = tokenizer.token_to_id(END_OF_BOOK)
    model.eval()
    try:
        for _ in range(max_new_tokens):
            context = input_ids[:, -model.config.max_sequence_length :]
            logits, _ = model(context)
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                threshold = torch.topk(
                    next_logits, min(top_k, next_logits.size(-1))
                ).values[:, -1:]
                next_logits = next_logits.masked_fill(
                    next_logits < threshold, float("-inf")
                )
            next_token = torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if end_id is not None and int(next_token.item()) == end_id:
                break
    finally:
        model.train(was_training)
    return tokenizer.decode(input_ids[0].tolist())
