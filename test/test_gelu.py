import unittest
import torch
import torch.nn as nn

from model.gelu import GELU


class TestGELU(unittest.TestCase):
    def test_gelu_basic(self):
        gelu = GELU()
        x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        out = gelu(x)
        expected = nn.GELU(approximate="tanh")(x)
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)

    def test_gelu_zero(self):
        gelu = GELU()
        x = torch.tensor([0.0])
        out = gelu(x)
        self.assertAlmostEqual(out.item(), 0.0, places=6)

    def test_gelu_positive(self):
        gelu = GELU()
        x = torch.tensor([10.0])
        out = gelu(x)
        self.assertAlmostEqual(out.item(), 10.0, places=4)

    def test_gelu_negative(self):
        gelu = GELU()
        x = torch.tensor([-10.0])
        out = gelu(x)
        self.assertAlmostEqual(out.item(), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
