import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize


class BrownCorpusDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=50):
        # Load content from file
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Tokenize content into sentences
        sentences = sent_tokenize(content)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Tokenize sentences dynamically and store encoded tokens
        self.encoded_sentences = self.tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.encoded_sentences["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encoded_sentences["input_ids"][idx],
            "attention_mask": self.encoded_sentences["attention_mask"][idx],
        }


def get_dataloader(
    file_path, batch_size=4, max_length=50, val_split=0.2, shuffle=True, seed=42
):
   
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = BrownCorpusDataset(file_path, tokenizer, max_length)

    # Split dataset into training and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, val_loader
