import unittest

import torch

from mogo.model.gpt import (
    GPT,
    GPTConfig,
    _apply_rope,
    _apply_rope_cached,
    _build_rope_cache,
    count_parameters,
)
from mogo.train.trainer import _optimizer_parameter_groups


class ModelTests(unittest.TestCase):
    def test_decoder_shapes_tied_weights_and_parameter_count(self):
        model = GPT(
            GPTConfig(
                vocab_size=64,
                hidden_size=32,
                num_layers=2,
                num_heads=4,
                intermediate_size=64,
                max_sequence_length=16,
                gradient_checkpointing=False,
            )
        )
        input_ids = torch.randint(0, 64, (2, 8))
        logits, loss = model(input_ids, input_ids)
        self.assertEqual(logits.shape, (2, 8, 64))
        self.assertIsNotNone(loss)
        self.assertIs(model.token_embedding.weight, model.lm_head.weight)
        self.assertEqual(
            count_parameters(model),
            sum(parameter.numel() for parameter in model.parameters()),
        )

    def test_cached_rope_matches_original_function(self):
        torch.manual_seed(5)
        values = torch.randn(2, 4, 7, 8)
        cos, sin = _build_rope_cache(7, 8, 10000.0, values.device)
        torch.testing.assert_close(
            _apply_rope_cached(values, cos, sin),
            _apply_rope(values, 10000.0),
            rtol=0,
            atol=0,
        )

    def test_future_tokens_do_not_change_earlier_logits(self):
        torch.manual_seed(7)
        model = GPT(
            GPTConfig(
                vocab_size=32,
                hidden_size=16,
                num_layers=2,
                num_heads=2,
                intermediate_size=32,
                max_sequence_length=8,
                gradient_checkpointing=False,
            )
        )
        model.eval()
        input_ids = torch.randint(0, 32, (1, 8))
        changed_ids = input_ids.clone()
        changed_ids[0, 5] = (changed_ids[0, 5] + 1) % 32
        original_logits, _ = model(input_ids)
        changed_logits, _ = model(changed_ids)
        torch.testing.assert_close(
            original_logits[:, :5], changed_logits[:, :5], rtol=0, atol=0
        )
        self.assertNotIn("rope_cos", model.state_dict())
        self.assertNotIn("rope_sin", model.state_dict())

    def test_gradient_checkpointing_backpropagates_with_rope_cache(self):
        model = GPT(
            GPTConfig(
                vocab_size=32,
                hidden_size=16,
                num_layers=2,
                num_heads=2,
                intermediate_size=32,
                max_sequence_length=8,
                gradient_checkpointing=True,
            )
        )
        input_ids = torch.randint(0, 32, (1, 8))
        _, loss = model(input_ids, input_ids)
        self.assertIsNotNone(loss)
        loss.backward()
        self.assertIsNotNone(model.blocks[0].attention.qkv.weight.grad)

    def test_overfit_tiny_batch(self):
        torch.manual_seed(3)
        model = GPT(
            GPTConfig(
                vocab_size=16,
                hidden_size=32,
                num_layers=1,
                num_heads=4,
                intermediate_size=64,
                max_sequence_length=8,
                gradient_checkpointing=False,
            )
        )
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model.train()
        _, initial_loss = model(tokens[:, :-1], tokens[:, 1:])
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(tokens[:, :-1], tokens[:, 1:])
            loss.backward()
            optimizer.step()
        _, final_loss = model(tokens[:, :-1], tokens[:, 1:])
        self.assertLess(float(final_loss.detach()), float(initial_loss.detach()) * 0.1)

    def test_optimizer_groups_exclude_embedding_and_norms(self):
        model = GPT(
            GPTConfig(
                vocab_size=64,
                hidden_size=32,
                num_layers=2,
                num_heads=4,
                intermediate_size=64,
                max_sequence_length=16,
                gradient_checkpointing=False,
            )
        )
        groups = _optimizer_parameter_groups(model, 0.1)
        decay_ids = {id(parameter) for parameter in groups[0]["params"]}
        no_decay_ids = {id(parameter) for parameter in groups[1]["params"]}
        self.assertEqual(groups[0]["weight_decay"], 0.1)
        self.assertEqual(groups[1]["weight_decay"], 0.0)
        self.assertNotIn(id(model.token_embedding.weight), decay_ids)
        self.assertIn(id(model.token_embedding.weight), no_decay_ids)
        for module in model.modules():
            if module.__class__.__name__ == "RMSNorm":
                self.assertIn(id(module.weight), no_decay_ids)
        self.assertFalse(decay_ids & no_decay_ids)
        self.assertEqual(
            len(decay_ids) + len(no_decay_ids),
            len(list(model.parameters())),
        )


if __name__ == "__main__":
    unittest.main()
