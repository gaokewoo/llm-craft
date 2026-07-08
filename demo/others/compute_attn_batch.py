import torch
from tools.log import log

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

#计算注意力得分
attn_scores = inputs @ inputs.T
log.info(attn_scores)

#对注意力得分进行softmax归一化
attn_weights = torch.softmax(attn_scores, dim=-1)
log.info(attn_weights)

#计算上下文向量
all_context_vecs = attn_weights @ inputs
log.info(all_context_vecs)
