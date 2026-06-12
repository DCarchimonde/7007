import re
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def preprocess_financial_text(text: str) -> str:
    """
    Strategic Entity Mask