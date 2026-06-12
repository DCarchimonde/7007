# WQF7007 Group25 - Task III Development and Demonstration

## Project
Financial Risk & Sentiment Analytics Engine for Sustainable Economic Monitoring, aligned with SDG 8: Decent Work and Economic Growth.

## Core NLP task
Sentence-level financial sentiment classification into three labels:

- Negative / Financial Risk
- Neutral
- Positive / Growth

## Dataset
Financial PhraseBank, `sentences_allagree` subset, loaded through Hugging Face datasets.

## Main components

- `dataset.py`: loads Financial PhraseBank, applies Strategic Entity Masking (SEM), and creates train/validation dataloaders.
- `model.py`: implements FinBERT + LoRA + BiLSTM + Custom Attention classification model.
- `train.py`: trains the final model with class-weighted Focal Loss.
- `run_ablation.py`: runs the ablation study comparing baseline, architecture, focal loss, and LoRA configurations.
- `app.py`: launches the Gradio demonstration interface with sentiment probabilities and attention heatmap visualization.

## How to run

```bash
pip install -r requirements.txt
python train.py
python run_ablation.py
python app.py
```

The Gradio app loads `finbert_bilstm_epoch_3.pt` if the trained weight file is present in the project root. If the file is missing, the interface will still launch, but predictions will not represent the trained final model.

## Submission note
For the final LMS submission, include:

1. Task III presentation slides PDF.
2. Task III final report PDF.
3. This GitHub repository link as the source code resource.
4. Dataset link or note: Hugging Face `financial_phrasebank`, `sentences_allagree` subset.
5. A Gradio screenshot or demo link showing the running application.
