import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


def preprocess_financial_text(text: str) -> str:
    """
    Strategic Entity Masking (SEM) used consistently in both training and demo inference.

    The goal is not to remove sentiment-bearing words, but to reduce memorisation of
    specific companies, ticker