import torch

from model.feed_forward import FeedForward
from model.config import GPT_CONFIG_124M
from tools.log import log

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
log.info(out.shape)
