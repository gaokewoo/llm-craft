import tiktoken
from tools.log import log

tokenizer = tiktoken.get_encoding("gpt2")
log.info("%s", tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None,
                 pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # 文本分词
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

        # 如果序列长度超过 max_length，则进行截断
        self.encoded_texts = [
            encoded_text[:self.max_length]
            for encoded_text in self.encoded_texts
        ]

        # 填充到最长序列的长度
        self.encoded_texts = [
            encoded_text + [pad_token_id] *
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

def get_spam_dataset(tokenizer, max_length=None,
                 pad_token_id=50256):
    train_dataset = SpamDataset(
        csv_file="data/output/train.csv",
        max_length=None,
        tokenizer=tokenizer
    )

    val_dataset = SpamDataset(
        csv_file="data/output/validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    test_dataset = SpamDataset(
        csv_file="data/output/test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )


    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

def main():
    train_loader, val_loader, test_loader = get_spam_dataset(tokenizer)
    for input_batch, target_batch in train_loader:
        pass
    log.info("Input batch dimensions: %s", input_batch.shape)
    log.info("Label batch dimensions %s", target_batch.shape)

    log.info(f"{len(train_loader)} training batches")
    log.info(f"{len(val_loader)} validation batches")
    log.info(f"{len(test_loader)} test batches")

if __name__ == "__main__":
    main()
