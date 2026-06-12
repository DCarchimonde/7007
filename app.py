import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import FinBERT_BiLSTM_Attention

# ==========================================
# 1. 初始化模型与加载权重
# ==========================================
print("Loading Model and Tokenizer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

model = FinBERT_BiLSTM_Attention(num_classes=3).to(device)

try:
    model.load_state_dict(torch.load("finbert_bilstm_epoch_3.pt", map_location=device))
    print("Weights loaded successfully!")
except FileNotFoundError:
    print("Warning: Weight file not found. Please ensure 'finbert_bilstm_epoch_3.pt' exists.")
    
model.eval()
labels_map = {0: "Negative (Financial Risk)", 1: "Neutral", 2: "Positive (Growth)"}

# ==========================================
# 2. 注入高级 CSS 样式 (修复溢出，提升质感)
# ==========================================
custom_css = """
.gradio-container { font-family: 'Inter', sans-serif; }
#title-section { text-align: center; margin-bottom: 20px; padding-bottom: 20px; border-bottom: 1px solid #e2e8f0; }
#title-section h1 { color: #1e3a8a; font-weight: 800; font-size: 2.5em; margin-bottom: 5px; }
#title-section h3 { color: #64748b; font-weight: 400; margin-top: 0; }
.heatmap-box {
    line-height: 2.4;
    font-size: 1.1rem;
    padding: 24px;
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.03);
    word-break: break-word; /* 完美解决文字溢出 */
    white-space: pre-wrap;
    color: #334155;
}
.token-badge {
    padding: 4px 6px;
    margin: 2px 1px;
    border-radius: 6px;
    display: inline-block;
    transition: all 0.2s ease;
}
.token-badge:hover { transform: scale(1.05); cursor: pointer; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
"""

# ==========================================
# 3. 核心推理与热力图渲染逻辑
# ==========================================
def predict_sentiment(text):
    if not text.strip():
        return {}, "<div class='heatmap-box'>Please enter a valid financial news text.</div>"

    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        logits, attn_weights = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        prob_dict = {labels_map[i]: float(probs[i]) for i in range(3)}
        attn = attn_weights.squeeze().cpu().numpy()
        
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
    
    html_out = "<div class='heatmap-box'>"
    for token, weight in zip(tokens, attn):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        alpha = min(weight * 5.0, 1.0) 
        bg_color = f"rgba(225, 29, 72, {alpha})" # Tailwind 极光红
        
        # 动态字体颜色：如果背景太红，字就变成白色
        text_color = "#ffffff" if alpha > 0.5 else "#0f172a"
        
        if token.startswith("##"):
            clean_token = token.replace("##", "")
            # 子词无缝拼接，没有左右 margin
            html_out += f"<span style='background-color: {bg_color}; color: {text_color}; padding: 4px 1px;' class='token-badge'>{clean_token}</span>"
        else:
            clean_token = token
            html_out += f" <span style='background-color: {bg_color}; color: {text_color};' class='token-badge'>{clean_token}</span>"
            
    html_out += "</div>"
    return prob_dict, html_out

# ==========================================
# 4. 构建企业级前端布局
# ==========================================
theme = gr.themes.Default(primary_hue="indigo", neutral_hue="slate").set(
    body_background_fill="#f1f5f9",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e2e8f0",
    block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)"
)

with gr.Blocks(theme=theme, css=custom_css) as demo:
    with gr.Column(elem_id="title-section"):
        gr.Markdown("# 📈 Financial Risk & Sentiment Analytics")
        gr.Markdown("### Universiti Malaya (UM) WQF7007 | Powered by FinBERT + BiLSTM + LoRA + Focal Loss")
    
    with gr.Row():
        # 左侧控制面板
        with gr.Column(scale=5):
            gr.Markdown("### 📥 Input Panel")
            text_input = gr.Textbox(
                lines=6, 
                show_label=False,
                placeholder="Paste financial news excerpt here...\n\nExample:\n'The company's operating profit plummeted by 50% due to severe supply chain disruptions, triggering a massive sell-off in early trading.'"
            )
            submit_btn = gr.Button("🚀 Run AI Analysis", variant="primary", size="lg")
            
            gr.Markdown("#### Quick Load Examples")
            gr.Examples(
                examples=[
                    "The company's operating profit plummeted by 50% due to severe supply chain disruptions.",
                    "Nokia expects revenue to remain stable in the upcoming quarter despite market volatility.",
                    "Apple announced a record-breaking dividend payout following massive iPhone sales surge in Asia."
                ],
                inputs=text_input
            )
            
        # 右侧结果面板
        with gr.Column(scale=7):
            gr.Markdown("### 📊 AI Output & Decision Transparency")
            with gr.Group():
                label_output = gr.Label(label="Probability Distribution (Softmax Confidence)")
            
            gr.Markdown("### 🔍 Temporal Attention Heatmap")
            gr.Markdown("*This visualization extracts the internal BiLSTM attention weights. Words highlighted in **crimson red** represent the primary lexical drivers for the model's prediction.*")
            html_output = gr.HTML()
            
    submit_btn.click(
        fn=predict_sentiment, 
        inputs=text_input, 
        outputs=[label_output, html_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006, share=True)