import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 使用一个线性层来组合头的输出
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 张量形状: (b, num_tokens, d_out)

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )  # 通过添加一个num_heads维度来隐式地分隔矩阵，然后展开最后一个维度; (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)  # 从形状(b, num_tokens, num_heads, head_dim)转换到(b, num_heads, num_tokens, head_dim)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # 被截断为词元数量的掩码

        attn_scores.masked_fill_(mask_bool, -torch.inf)  # 计算每个头的点积

        attn_weights = torch.softmax(  # 使用掩码来填充注意力分数
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )  # 张量形状:(b, num_tokens, n_numheads, head_dim)

        context_vec = self.out_proj(context_vec)  # 添加一个可选的线性投影
        return context_vec  # 组合头，其中self.d_out = self.num_heads * self.head_dim
