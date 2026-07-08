import torch
from model.gpt_model import GPTModel
from llm_example.gpt_example.config import GPT_CONFIG_124M
from tools.generate_text import text_to_token_ids, token_ids_to_text, generate
import tiktoken
from tools.log import log

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("data/output/model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
log.info("Output text:\n%s", token_ids_to_text(token_ids, tokenizer))
