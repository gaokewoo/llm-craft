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

#初始化权重矩阵
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

#计算x_2的查询向量
x_2 = inputs[1]
query_2 = x_2 @ W_query
log.info(query_2)

# 计算所有的键和值向量
keys = inputs @ W_key
values = inputs @ W_value
log.info("keys.shape: %s", keys.shape)
log.info("values.shape: %s", values.shape)

#计算注意力得分
attn_scores_2 = query_2 @ keys.T
log.info(attn_scores_2)

#对注意力得分进行softmax归一化
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
log.info(attn_weights_2)

#计算上下文向量
context_vec_2 = attn_weights_2 @ values
log.info(context_vec_2)
