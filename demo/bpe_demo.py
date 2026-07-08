from importlib.metadata import version

import tiktoken
from tools.log import log

log.info("tiktoken version: %s", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
log.info(integers)

# 将每个 token id 与其对应的文本片段一一对应
decoded_tokens = tokenizer.decode_tokens_bytes(integers)
for idx, (token_id, token_bytes) in enumerate(zip(integers, decoded_tokens)):
    token_str = token_bytes.decode("utf-8", errors="replace")
    log.info("[%2d] %6d -> %r", idx, token_id, token_str)


with open("data/output/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
log.info(len(enc_text))


enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
log.info("x: %s", x)
log.info("y:      %s", y)

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    log.info("%s ----> %s", context, desired)

for i in range(1, context_size+1):
   context = enc_sample[:i]
   desired = enc_sample[i]
   log.info("%s ----> %s", tokenizer.decode(context), tokenizer.decode([desired]))

