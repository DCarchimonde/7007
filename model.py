import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType

class CustomAttention(nn.Module):
    """
    我们在 Proposal 中承诺的 Saliency Attention Layer (Phase I & II 融合点)
    提取特征权重，既用于分类，也用于后期的 Gradio 热力图展示。
    """
    def __init__(self, hidden_size):
        super(CustomAttention, self).__init__()
        # 一个简单的线性映射来计算每个时间步的重要性
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output, mask):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        # mask shape: (batch_size, seq_len)
        
        # 计算注意力分数 (batch_size, seq_len, 1)
        attn_scores = self.attention(lstm_output)
        attn_scores = attn_scores.squeeze(-1) # (batch_size, seq_len)
        
        # 把 mask 为 0 的地方 (padding部分) 的注意力分数设为极小值
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax 归一化，得到权重 alpha_i (我们在 Proposal 里提到的)
        attn_weights = torch.softmax(attn_scores, dim=-1) # (batch_size, seq_len)
        
        # 加权求和得到句子级别的特征表示 (batch_size, hidden_size)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        
        return context_vector, attn_weights


class FinBERT_BiLSTM_Attention(nn.Module):
    def __init__(self, num_classes=3, lora_r=8, lora_alpha=16, lstm_hidden=256):
        super(FinBERT_BiLSTM_Attention, self).__init__()
        
        print("Loading base FinBERT model...")
        # 加载裸的 FinBERT (不带分类头)
        self.finbert = AutoModel.from_pretrained("ProsusAI/finbert")
        
        # === 核心贡献 1: 引入 LoRA ===
        # 我们在 Proposal 4.1.2 节承诺：冻结原始权重，只更新低秩矩阵
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            inference_mode=False, 
            r=lora_r, 
            lora_alpha=lora_alpha, 
            lora_dropout=0.1,
            target_modules=["query", "value"] # 针对 Transformer 的 Q 和 V 层注入
        )
        self.finbert = get_peft_model(self.finbert, peft_config)
        self.finbert.print_trainable_parameters() # 打印出来会发现可训练参数不到 1%
        
        finbert_hidden_size = self.finbert.config.hidden_size # 通常是 768
        
        # === 核心贡献 2: Temporal Saliency via BiLSTM ===
        # 我们在 Proposal 3.3.2 节写下的双向 LSTM
        self.bilstm = nn.LSTM(
            input_size=finbert_hidden_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 双向的话，特征维度翻倍
        lstm_out_dim = lstm_hidden * 2
        
        # === 核心贡献 3: Saliency Attention Layer ===
        self.attention = CustomAttention(lstm_out_dim)
        
        # 分类头 (3 分类: Neg, Neu, Pos)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        # 1. 过 FinBERT + LoRA
        # 输出的 last_hidden_state 包含了所有 token 的特征 (batch, seq_len, 768)
        outputs = self.finbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # 2. 过 BiLSTM
        # lstm_out 形状: (batch, seq_len, lstm_hidden*2)
        lstm_out, _ = self.bilstm(sequence_output)
        
        # 3. 过 Attention 机制
        # context_vector 是加权后的句子特征，attn_weights 是我们要画热力图的东西
        context_vector, attn_weights = self.attention(lstm_out, attention_mask)
        
        # 4. 最终分类
        logits = self.classifier(context_vector)
        
        return logits, attn_weights

# --- 测试模型架构 ---
if __name__ == "__main__":
    from dataset import get_dataloaders
    
    # 拿一小批数据试试毒
    train_loader, _, _ = get_dataloaders(batch_size=4)
    batch = next(iter(train_loader))
    
    # 初始化模型
    model = FinBERT_BiLSTM_Attention()
    
    # 丢到 GPU 上跑跑看
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # 前向传播
    print("\n--- Model Forward Test ---")
    logits, attn_weights = model(input_ids, attention_mask)
    
    print("Logits Shape (Should be [4, 3]):", logits.shape)
    print("Attention Weights Shape (Should be [4, 128]):", attn_weights.shape)
    print("Success! The Hybrid Architecture is ready.")