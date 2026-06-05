import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class FinancialPhraseBankDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Strategic Entity Masking (SEM) - 兑现我们在 Proposal 里的承诺
        # 将数字替换为 [NUMBER]
        text = re.sub(r'\b\d+(?:\.\d+)?\b', '[NUMBER]', text)
        # 简单模拟公司名掩码 (如果是全大写单词，比如 NOKIA，或者结尾是 Inc/Corp)
        text = re.sub(r'\b([A-Z]+|.*? (Inc\.|Corp\.|Ltd\.|Oyj))\b', '[COMPANY]', text)

        # 编码
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_dataloaders(batch_size=32):
    print("Downloading/Loading Financial PhraseBank dataset...")
    # 'sentences_allagree' 是最严格的数据子集
    dataset = load_dataset("financial_phrasebank", "sentences_allagree", split='train', trust_remote_code=True)
    
    # 标签映射: 0: Negative, 1: Neutral, 2: Positive
    texts = dataset['sentence']
    labels = dataset['label']
    
    # 加载 FinBERT 的 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    
    # 切分训练集和验证集 (80/20)
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels # stratify 保证极度不平衡的数据被均匀划分
    )
    
    train_dataset = FinancialPhraseBankDataset(train_texts, train_labels, tokenizer)
    val_dataset = FinancialPhraseBankDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, tokenizer

# --- 测试一下咱们的地基打好没 ---
if __name__ == "__main__":
    train_loader, val_loader, tokenizer = get_dataloaders(batch_size=8)
    batch = next(iter(train_loader))
    print("\n--- Data Check ---")
    print("Input IDs shape:", batch['input_ids'].shape)
    print("Labels shape:", batch['label'].shape)
    print("Sample label:", batch['label'][0].item())
    
    # 看看咱们的 Masking 生效没
    decoded_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
    print("\nSample Processed Text (Check for [COMPANY] and [NUMBER]):")
    print(decoded_text)