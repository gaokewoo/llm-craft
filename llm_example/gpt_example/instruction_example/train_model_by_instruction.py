import tiktoken
import torch
import json

from llm_example.gpt_example.instruction_example.instruction_tools import format_input
from llm_example.gpt_example.instruction_example.instruction_dataset import get_instruction_dataset
from llm_example.gpt_example.gpt_model.load_gpt_model import load_gpt_model
from tools.generate_text import generate, text_to_token_ids, token_ids_to_text
from tools.model_tool import calc_loss_loader
from tools.log import log
from model.trainer import train_model_simple

tokenizer = tiktoken.get_encoding("gpt2")
log.info("%s", tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

num_workers = 0  # 如果你的操作系统支持 Python 进程的并行，那么可以加大这个数值
batch_size = 8
torch.manual_seed(123)

train_loader, val_loader, test_loader = get_instruction_dataset(tokenizer, num_workers, batch_size)
train_data = train_loader.dataset.data
val_data = val_loader.dataset.data
test_data = test_loader.dataset.data

log.info("Train loader:")
for inputs, targets in train_loader:
    log.info("%s %s", inputs.shape, targets.shape)

input_text = format_input(val_data[0])
log.info("%s", input_text)

CHOOSE_MODEL = "gpt2-medium (355M)"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, config = load_gpt_model(device, CHOOSE_MODEL)

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=config["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)
log.info("%s", generated_text)
response_text = generated_text[len(input_text):].strip()
log.info("%s", response_text)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=5
)

log.info("Training loss: %s", train_loss)
log.info("Validation loss: %s", val_loss)

import time

start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.00005, weight_decay=0.1
)
num_epochs = 2

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
log.info(f"Training completed in {execution_time_minutes:.2f} minutes.")

from tools.matplot import plot_values
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_values(epochs_tensor, tokens_seen, train_losses, val_losses)

torch.manual_seed(123)

for entry in test_data[:3]:  # 遍历前 3 个测试样本
    input_text = format_input(entry)
    token_ids = generate(  # 使用 7.5 节中引入的生成函数
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=config["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    log.info("%s", input_text)
    log.info(f"\nCorrect response:\n>> {entry['output']}")
    log.info(f"\nModel response:\n>> {response_text.strip()}")
    log.info("-----------------------------------")

from tqdm import tqdm

for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=config["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)

    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    test_data[i]["model_response"] = response_text

with open("data/output/instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)

log.info("%s", test_data[0])

import re

file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
torch.save(model.state_dict(), "data/output/"+file_name)
log.info(f"Model saved as {file_name}")
