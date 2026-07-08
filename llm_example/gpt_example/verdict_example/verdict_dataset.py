import torch
import tiktoken

from tools.log import log
from demo.others.dataloader_v1 import create_dataloader_v1
from llm_example.gpt_example.config import GPT_CONFIG_124M

def get_verdict_dataset(file_path= "data/output/the-verdict.txt"):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    tokenizer = tiktoken.get_encoding("gpt2")

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    log.info("Characters:%s", total_characters)
    log.info("Tokens:%s", total_tokens)

    # 将输入文本分割为训练集和验证集
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=int(GPT_CONFIG_124M["context_length"]),
        stride=int(GPT_CONFIG_124M["context_length"]),
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=int(GPT_CONFIG_124M["context_length"]),
        stride=int(GPT_CONFIG_124M["context_length"]),
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader

def main():
    train_loader, val_loader = get_verdict_dataset()
    log.info("Train loader:")
    for x, y in train_loader:
        log.info("%s %s", x.shape, y.shape)

    log.info("Validation loader:")
    for x, y in val_loader:
        log.info("%s %s", x.shape, y.shape)


if __name__ == "__main__":
    main()