
import torch
import tiktoken
from model.gpt_model import GPTModel
from llm_example.gpt_example.verdict_example.verdict_dataset import get_verdict_dataset
from llm_example.gpt_example.config import GPT_CONFIG_124M
from model.trainer import train_model_simple,save_model   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")

train_loader, val_loader = get_verdict_dataset()

model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

save_model(model,optimizer, "data/output/model_and_optimizer.pth")
