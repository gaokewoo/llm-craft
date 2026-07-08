import torch
import torch.nn as nn
from tools.log import log

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)  # register_buffer调用也是一个新版本（下文提供了更多信息）
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape          # 将维度1和2转置，
        keys = self.W_key(x)                    # 将批维度保持在第一个位置（0）
        queries = self.W_query(x)
        values = self.W_value(x)
        log.info("keys:\n%s", keys)
        log.info("queries:\n%s", queries)
        log.info("values:\n%s", values)


        attn_scores = queries @ keys.transpose(1, 2)
        log.info("attn_scores:\n%s", attn_scores)

        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        log.info("attn_scores masked fill:\n%s", attn_scores)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )                                        # 在PyTorch中，带有尾随下划线的操作将就
                                                   # 地执行，从而避免了不必要的内存副本
        log.info("attn_weights:\n%s", attn_weights)

        attn_weights = self.dropout(attn_weights)
        log.info("attn_weights dropout:\n%s", attn_weights)

        context_vec = attn_weights @ values
        log.info("context_vec:\n%s", context_vec)
        
        return context_vec


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
context_length = batch.shape[1]
ca = CausalAttention(d_in=3, d_out=2, context_length=context_length, dropout=0.0)
context_vecs = ca(batch)
log.info(context_vecs)
log.info("context_vecs.shape: %s", context_vecs.shape)

