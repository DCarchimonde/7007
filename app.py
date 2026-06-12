import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from dataset import preprocess_financial_text
from model import FinBERT_BiLSTM_Attention


device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = FinBERT_BiLSTM_Attention(num_classes=3).to(device)

try:
    state_dict = torch.load("finbert_bilstm_epoch_3.pt", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("Loaded finbert_bilstm_epoch_3.pt successfully.")
except Exception as e:
    print(f"Warning: could not load trained weights: {e}")

model.eval()
labels_map = {0: "Negative (Financial Risk)", 1: "Neutral", 2: "Positive (Growth)"}

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
    word-break: break-word;
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


def predict_sentiment(text):
    if not text or not text.strip():
        return {}, "<div class='heatmap-box'>Please enter a valid financial news text.</div>"

    processed_text = preprocess_financial_text(text)
    inputs = tokenizer(processed_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits, attn_weights = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        prob_dict = {labels_map[i]: float(probs[i]) for i in range(3)}
        attn = attn_weights.squeeze().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())

    html_out = "<div class='heatmap-box'>"
    html_out += "<div style='font-size:0.9rem; color:#64748b; margin-bottom:10px;'>SEM text: " + processed_text + "</div>"
    for token, weight in zip(tokens, attn):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        alpha = min(float(weight) * 5.0, 1.0)
        bg_color = f"rgba(225, 29, 72, {alpha})"
        text_color = "#ffffff" if alpha > 0.5 else "#0f172a"

        if token.startswith("##"):
            clean_token = token.replace("##", "")
            html_out += f"<span style='background-color: {bg_color}; color: {text_color}; padding: 4px 1px;' class='token-badge'>{clean_token}</span>"
        else:
            clean_token = token
            html_out += f" <span style='background-color: {bg_color}; color: {text_color};' class='token-badge'>{clean_token}</span>"

    html_out += "</div>"
    return prob_dict, html_out


theme = gr.themes.Default(primary_hue="indigo", neutral_hue="slate").set(
    body_background_fill="#f1f5f9",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e2e8f0",
    block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
)

with gr.Blocks(theme=theme, css=custom_css) as demo:
    with gr.Column(elem_id="title-section"):
        gr.Markdown("# Financial Risk & Sentiment Analytics")
        gr.Markdown("### Universiti Malaya WQF7007 | FinBERT + BiLSTM + Attention + LoRA + Focal Loss")

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### Input Panel")
            text_input = gr.Textbox(
                lines=6,
                show_label=False,
                placeholder="Paste financial news excerpt here...\n\nExample:\nThe company's operating profit plummeted by 50% due to severe supply chain disruptions.",
            )
            submit_btn = gr.Button("Run AI Analysis", variant="primary", size="lg")

            gr.Markdown("#### Quick Load Examples")
            gr.Examples(
                examples=[
                    "The company's operating profit plummeted by 50% due to severe supply chain disruptions.",
                    "Nokia expects revenue to remain stable in the upcoming quarter despite market volatility.",
                    "Apple announced a record-breaking dividend payout following massive iPhone sales surge in Asia.",
                ],
                inputs=text_input,
            )

        with gr.Column(scale=7):
            gr.Markdown("### AI Output & Decision Transparency")
            label_output = gr.Label(label="Probability Distribution (Softmax Confidence)")
            gr.Markdown("### Temporal Attention Heatmap")
            gr.Markdown("Words highlighted in crimson red represent the primary lexical drivers for the model prediction.")
            html_output = gr.HTML()

    submit_btn.click(fn=predict_sentiment, inputs=text_input, outputs=[label_output, html_output])

if __name__ == "__main__":
    demo.launch()
