import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, accuracy_score
import numpy as np
import random
import os

from dataset import get_dataloaders
from model import FinBERT_BiLSTM_Attention


def set_global_seeds(seed=42):
    """Fix random seeds for reproducible ablation results."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global Random Seed set to {seed} for deterministic execution.")


class FocalLoss(nn.Module):
    """
    Multi-class focal loss:
        FL = - alpha_t * (1 - p_t)^gamma * log(p_t)

    Important implementation detail: p_t is computed from the raw cross-entropy
    without class weights. The alpha term is applied afterwards, matching the
    mathematical formula reported in the Task III report.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_factor = (1 - pt).pow(self.gamma)
        loss = -focal_factor * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def train_and_evaluate(epochs=3, batch_size=32, lr=2e-4):
    set_global_seeds(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)
    model = FinBERT_BiLSTM_Attention().to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    alpha_weights = torch.tensor([5.0, 1.0, 2.5], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)

    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / max(1, len(train_loader))

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
            for batch in val_loop:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits, _ = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        neg_recall = recall_score(all_labels, all_preds, labels=[0], average=None, zero_division=0)[0]

        print(f"\n--- Epoch {epoch + 1} Results ---")
        print(f"Train Loss:  {avg_train_loss:.4f}")
        print(f"Accuracy:    {acc:.4f}")
        print(f"Macro-F1:    {macro_f1:.4f}")
        print(f"Neg Recall:  {neg_recall:.4f}")
        print("-" * 30)

        torch.save(model.state_dict(), f"finbert_bilstm_epoch_{epoch + 1}.pt")


if __name__ == "__main__":
    train_and_evaluate(epochs=3, batch_size=32, lr=2e-4)
