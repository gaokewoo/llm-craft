import torch
from torch import nn

from model.layer_norm import LayerNorm
from tools.log import log

torch.manual_seed(123)
torch.set_printoptions(sci_mode=False)
batch_example = torch.randn(2, 5)
log.info(batch_example)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
log.info(out)

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
log.info("Mean:\n%s", mean)
log.info("Variance:\n%s", var)

