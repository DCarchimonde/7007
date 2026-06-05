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

# ==========================================
# 动态架构构建器 (一键切换 4 种实验形态)
# ==========================================
class DynamicFinBERT(nn.Module):
    def __init__(self, use_bilstm=False, use_lora=False, num_classes=3):
        super(DynamicFinBERT, self).__init__()
        self.use_bilstm = use_bilstm
        self.use_lora = use_lora
        
        self.finbert = AutoModel.from_pretrained("ProsusAI/finbert")
        
        if self.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, 
                r=8, lora_alpha=16, lora_dropout=0.1,
                target_modules=["query", "value"]
            )
            self.finbert = get_peft_model(self.finbert, peft_config)
            
        hidden_size = self.finbert.config.hidden_size
        
        if self.use_bilstm:
            lstm_hidden = 256
            self.bilstm = nn.LSTM(hidden_size, lstm_hidden, batch_first=True, bidirectional=True)
            self.attention = CustomAttention(lstm_hidden * 2)
            clf_in = lstm_hidden * 2
        else:
            # Baseline: 直接用 CLS token
            clf_in = hidden_size
            
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(clf_in, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.finbert(input_ids=input_ids, attention_mask=attention_mask)
        
        if self.use_bilstm:
            seq_out = outputs.last_hidden_state
            lstm_out, _ = self.bilstm(seq_out)
            context, _ = self.attention(lstm_out, attention_mask)
        else:
            # Baseline 获取 CLS token (第一个 token)
            context = outputs.last_hidden_state[:, 0, :]
            
        return self.classifier(context)

# ==========================================
# 单个实验运行器
# ==========================================
def run_single_experiment(exp_name, use_bilstm, use_lora, use_focal_loss, train_loader, val_loader, device):
    set_global_seeds(42)
    print(f"\n{'='*50}\nStarting {exp_name}\n{'='*50}")
    
    model = DynamicFinBERT(use_bilstm=use_bilstm, use_lora=use_lora).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-4)
    
    if use_focal_loss:
        alpha_weights = torch.tensor([5.0, 1.0, 2.5]).to(device)
        criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)
    else:
        # Standard Cross Entropy
        criterion = nn.CrossEntropyLoss()
        
    # 为了快速消融，我们每个实验只跑 2 个 Epoch
    for epoch in range(2):
        model.train()
        for batch in tqdm(train_loader, desc=f"Train Ep{epoch+1}"):
            optimizer.zero_grad()
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['label'].numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    neg_recall = recall_score(all_labels, all_preds, labels=[0], average=None)[0]
    
    return {"Experiment": exp_name, "Accuracy": acc, "Macro-F1": macro_f1, "Neg-Recall": neg_recall}

# ==========================================
# 自动化消融实验矩阵执行
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Batch size 设小点，因为 Exp1 和 Exp2 没有 LoRA，全参数微调极度吃显存
    train_loader, val_loader, _ = get_dataloaders(batch_size=16) 
    
    results = []
    
    # 矩阵定义 [Exp_Name, use_bilstm, use_lora, use_focal_loss]
    experiments = [
        ("Exp 1 (Baseline)", False, False, False),
        ("Exp 2 (Architecture)", True, False, False),
        ("Exp 3 (Algorithmic)", True, False, True),
        ("Exp 4 (Efficiency)", True, True, True)
    ]
    
    for exp_name, u_bi, u_lo, u_fl in experiments:
        res = run_single_experiment(exp_name, u_bi, u_lo, u_fl, train_loader, val_loader, device)
        results.append(res)
        
    # 打印最终对比表格
    print("\n\n" + "="*60)
    print("FINAL ABLATION STUDY RESULTS")
    print("="*60)
    df = pd.DataFrame(results)
    # 格式化输出为 Markdown 表格，方便直接复制到你的 Report 里
    print(df.to_markdown(index=False, floatfmt=".4f"))