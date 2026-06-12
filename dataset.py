import re

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def preprocess_financial_text(text: str) -> str:
    """Strategic Entity Masking used for both training and Gradio inference."""
    text = str(text)
    text = re.sub(r"\b[A-Z]{1,5}\b", " [TICKER] ", text)
    text = re.sub(r"[$€£¥]?\b\d+(?:\.\d+)?\s?(?:%|bn|billion|m|million|k|thousand)?\b", " [NUM] ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[A-Z][A-Za-z&.-]*(?:\s+[A-Z][A-Za-z&.-]*)*\s+(?:Inc|Corp|Corporation|Ltd|Limited|PLC|Group|Holdings|Bank|Co)\b", " [ORG] ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class FinancialPhraseBankDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = [int(x) for x in labels]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            preprocess_financial_text(self.texts[idx]),
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _load_phrasebank_allagree():
    ds = load_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)["train"]
    texts = ds["sentence"]
    raw_labels = ds["label"]
    labels = [LABEL_MAP[x.lower()] if isinstance(x, str) else int(x) for x in raw_labels]
    return texts, labels


def get_dataloaders(batch_size=32, max_length=128, test_size=0.2, seed=42):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    texts, labels = _load_phrasebank_allagree()
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    train_ds = FinancialPhraseBankDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds = FinancialPhraseBankDataset(val_texts, val_labels, tokenizer, max_length)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        tokenizer,
    )
