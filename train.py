import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, accuracy_score
import numpy as np
import random
import os

# 导入咱们刚刚写好的模块
from dataset import get_dataloaders
from model import FinBERT_BiLSTM_Attention

# ==========================================
# 核心承诺 1: 锁死全局随机种子 (确保复现性)
# ==========================================
def set_global_seeds(seed=42):
    """
    固定所有引擎的随机种子，保证超参数调优结果可复现
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果用多卡
    # 保证 CuDNN 决定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global Random Seed set to {seed} for deterministic execution.")

# ==========================================
# 核心承诺 2: Focal Loss (解决 12.5% 极度不平衡)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # 类别权重 tensor
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes], targets: [batch_size]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss) # 获取模型对真实标签的预测概率 p_t
        focal_loss = ((1 - pt) ** self.gamma * ce_loss) # 核心公式: (1-p_t)^gamma * CE
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# ==========================================
# 核心训练循环
# ==========================================
def train_and_evaluate(epochs=3, batch_size=32, lr=2e-4):
    set_global_seeds(42) # 调用固定种子
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 1. 拿数据
    train_loader, val_loader, tokenizer = get_dataloaders(batch_size=batch_size)
    
    # 2. 拿模型
    model = FinBERT_BiLSTM_Attention().to(device)
    
    # 3. 设置优化器与 Focal Loss
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # 根据 Proposal 数据分布: Negative(0): 12.5%, Neutral(1): 59.4%, Positive(2): 28.1%
    # 赋予少数类 (Negative) 更高的权重
    alpha_weights = torch.tensor([5.0, 1.0, 2.5]).to(device) 
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
    
    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 训练阶段
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = total_loss / len(train_loader)
        
        # 验证阶段 (核心承诺 3: 严格的评估指标)
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in val_loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label']
                
                logits, _ = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                
        # 计算极其严谨的指标
        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        # labels=[0] 表示单独提取出 Negative 类的 Recall (我们最关心的风险信号)
        neg_recall = recall_score(all_labels, all_preds, labels=[0], average=None)[0] 
        
        print(f"\n--- Epoch {epoch+1} Results ---")
        print(f"Train Loss:  {avg_train_loss:.4f}")
        print(f"Accuracy:    {acc:.4f} (Deceptive metric due to imbalance)")
        print(f"Macro-F1:    {macro_f1:.4f} (True model equilibrium)")
        print(f"Neg Recall:  {neg_recall:.4f} (Ability to catch financial risks)")
        print("-" * 30)
        
        # 保存这轮权重
        torch.save(model.state_dict(), f"finbert_bilstm_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    # 为了跑得快先用 batch_size=32 跑 3 个 Epoch 试试水
    train_and_evaluate(epochs=3, batch_size=32, lr=2e-4)