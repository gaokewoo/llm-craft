import tiktoken
import torch

from model.gpt_model import GPTModel
from model.config import GPT_CONFIG_124M
from tools.log import log
from tools.generate_text import generate_text_simple

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
log.info(batch)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
log.info("Input batch:\n%s", batch)
log.info("Output shape: %s", out.shape)
log.info(out)

total_params = sum(p.numel() for p in model.parameters())
log.info("Total number of parameters: %d", total_params)


#原始GPT-2架构中使用了一个叫作权重共享(weight tying)的概念。
#也就是说，原始GPT-2架构是将词元嵌入层作为输出层重复使用的
log.info("Token embedding layer shape: %s", model.tok_emb.weight.shape)
log.info("Output layer shape: %s", model.out_head.weight.shape)

#由于分词器词汇表中有50 257个条目，因此词元嵌入层和输出层非常庞大。
#根据权重共享的概念，我们需要从总的GPT-2模型参数计数中减去输出层的参数量
total_params_gpt2 = (
    total_params - sum(p.numel()
    for p in model.out_head.parameters())
)
log.info("Number of trainable parameters considering weight tying: %d", total_params_gpt2)

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
log.info("encoded:%s", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
log.info("encoded_tensor.shape:%s", encoded_tensor.shape)

model.eval()
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
log.info("Output:%s", out)
log.info("Output length:%s", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
log.info(decoded_text)
