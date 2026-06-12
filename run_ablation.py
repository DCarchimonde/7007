import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import f1_score, recall_score, accuracy_score
from tqdm import tqdm
import pandas as pd

from dataset import get_dataloaders
from train import set_global_seeds, FocalLoss
from model import CustomAttention


class DynamicFinBERT(nn.Module):
    def __init__(self, use_bilstm=False, use_lora=False, num_classes=3):
        super().__init__()
        self.use_bilstm = use_bilstm
        self.use_lora = use_lora

        self.finbert = AutoModel.from_pretrained("ProsusAI/finbert")

        if self.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["query", "value"],
            )
            self.finbert = get_peft_model(self.finbert, peft_config)

        hidden_size = self.finbert.config.hidden_size

        if self.use_bilstm:
            lstm_hidden = 256
            self.bilstm = nn.LSTM(hidden_size, lstm_hidden, batch_first=True, bidirectional=True)
            self.attention = CustomAttention(lstm_hidden * 2)
            clf_in = lstm_hidden * 2
        else:
            clf_in = hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(clf_in, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.finbert(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_bilstm:
            seq_out = outputs.last_hidden_state
            lstm_out, _ = self.bilstm(seq_out)
            context, _ = self.attention(lstm_out, attention_mask)
        else:
            context = outputs.last_hidden_state[:, 0, :]
        return self.classifier(context)


def run_single_experiment(exp_name, use_bilstm, use_lora, use_focal_loss, train_loader, val_loader, device):
    set_global_seeds(42)
    print(f"\n{'=' * 50}\nStarting {exp_name}\n{'=' * 50}")

    model = DynamicFinBERT(use_bilstm=use_bilstm, use_lora=use_lora).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-4)

    if use_focal_loss:
        alpha_weights = torch.tensor([5.0, 1.0, 2.5], dtype=torch.float32).to(device)
        criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    for epoch in range(2):
        model.train()
        for batch in tqdm(train_loader, desc=f"{exp_name} Train Ep{epoch + 1}"):
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    neg_recall = recall_score(all_labels, all_preds, labels=[0], average=None, zero_division=0)[0]

    return {"Experiment": exp_name, "Accuracy": acc, "Macro-F1": macro_f1, "Neg-Recall": neg_recall}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataloaders(batch_size=16)

    experiments = [
        ("Exp 1 (Baseline)", False, False, False),
        ("Exp 2 (Architecture)", True, False, False),
        ("Exp 3 (Architecture + Focal Loss)", True, False, True),
        ("Exp 4 (Architecture + Focal Loss + LoRA)", True, True, True),
    ]

    results = [
        run_single_experiment(name, use_bilstm, use_lora, use_focal_loss, train_loader, val_loader, device)
        for name, use_bilstm, use_lora, use_focal_loss in experiments
    ]

    print("\n\n" + "=" * 60)
    print("FINAL ABLATION STUDY RESULTS")
    print("=" * 60)
    print(pd.DataFrame(results).to_markdown(index=False, floatfmt=".4f"))
