import unittest

import torch

from mogo.model.gpt import GPT, GPTConfig, count_parameters


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


if __name__ == "__main__":
    unittest.main()
