import torch

from tools.log import log
from model.gelu import GELU

gelu = GELU()
x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
out = gelu(x)
log.info(out)
