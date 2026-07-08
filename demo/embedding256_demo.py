import torch
from tools.log import log
from demo.others.dataloader_v1 import create_dataloader_v1

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

with open("data/output/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
   stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
log.info("Token IDs:\n%s", inputs)
log.info("Inputs shape:\n%s", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
log.info(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
log.info(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
log.info(input_embeddings.shape)

