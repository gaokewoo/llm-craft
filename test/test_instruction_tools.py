import unittest
from pathlib import Path

import torch

from llm_example.gpt_example.instruction_example.instruction_tools import custom_collate_draft_1, custom_collate_draft_2, customized_collate_fn

class TestInstructionTools(unittest.TestCase):
    PAD_TOKEN_ID = 50256

    def test_custom_collate_draft_1(self):
        inputs_1 = [0, 1, 2, 3, 4]
        inputs_2 = [5, 6]
        inputs_3 = [7, 8, 9]
        batch = (inputs_1, inputs_2, inputs_3)

        result = custom_collate_draft_1(batch, pad_token_id=self.PAD_TOKEN_ID)

        expected = torch.tensor([
            [0, 1, 2, 3, 4],
            [5, 6, self.PAD_TOKEN_ID, self.PAD_TOKEN_ID, self.PAD_TOKEN_ID],
            [7, 8, 9, self.PAD_TOKEN_ID, self.PAD_TOKEN_ID],
        ])
        torch.testing.assert_close(result, expected)

    def test_custom_collate_draft_2(self):
        inputs_1 = [0, 1, 2, 3, 4]
        inputs_2 = [5, 6]
        inputs_3 = [7, 8, 9]
        batch = (inputs_1, inputs_2, inputs_3)

        inputs, targets = custom_collate_draft_2(
            batch, pad_token_id=self.PAD_TOKEN_ID
        )

        expected_inputs = torch.tensor([
            [0, 1, 2, 3, 4],
            [5, 6, self.PAD_TOKEN_ID, self.PAD_TOKEN_ID, self.PAD_TOKEN_ID],
            [7, 8, 9, self.PAD_TOKEN_ID, self.PAD_TOKEN_ID],
        ])
        expected_targets = torch.tensor([
            [1, 2, 3, 4, self.PAD_TOKEN_ID],
            [6, self.PAD_TOKEN_ID, self.PAD_TOKEN_ID, self.PAD_TOKEN_ID, self.PAD_TOKEN_ID],
            [8, 9, self.PAD_TOKEN_ID, self.PAD_TOKEN_ID, self.PAD_TOKEN_ID],
        ])
        torch.testing.assert_close(inputs, expected_inputs)
        torch.testing.assert_close(targets, expected_targets)

    def test_customized_collate_fn(self):
        inputs_1 = [0, 1, 2, 3, 4]
        inputs_2 = [5, 6]
        inputs_3 = [7, 8, 9]
        batch = (inputs_1, inputs_2, inputs_3)

        inputs, targets = customized_collate_fn(
            batch, pad_token_id=self.PAD_TOKEN_ID, ignore_index=-100
        )

        expected_inputs = torch.tensor([
            [0, 1, 2, 3, 4],
            [5, 6, self.PAD_TOKEN_ID, self.PAD_TOKEN_ID, self.PAD_TOKEN_ID],
            [7, 8, 9, self.PAD_TOKEN_ID, self.PAD_TOKEN_ID],
        ])
        expected_targets = torch.tensor([
            [1, 2, 3, 4, self.PAD_TOKEN_ID],
            [6, self.PAD_TOKEN_ID, -100, -100, -100],
            [8, 9, self.PAD_TOKEN_ID, -100, -100],
        ])
        torch.testing.assert_close(inputs, expected_inputs)
        torch.testing.assert_close(targets, expected_targets)


if __name__ == "__main__":
    unittest.main()
