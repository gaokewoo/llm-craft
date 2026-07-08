import torch
from tools.log import log
from model.multi_head_attention import MultiHeadAttention

a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],  # 这个张量的形状是(b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4)
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],

                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])

log.info(a @ a.transpose(2, 3))

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

batch = torch.stack((inputs, inputs), dim=0)
log.info(batch.shape) # 2 inputs with 6 tokens each, and each token has embedding dimension 3

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
log.info(context_vecs)
log.info("context_vecs.shape: %s", context_vecs.shape)
