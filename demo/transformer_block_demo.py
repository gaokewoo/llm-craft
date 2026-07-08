import torch
from tools.log import log
from model.transformer_block import TransformerBlock
from model.config import GPT_CONFIG_124M

torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

log.info("Input shape: %s", x.shape)
log.info("Output shape: %s", output.shape)
