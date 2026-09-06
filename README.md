# Efficient Vision Transformer for Model Compression

A compact Vision Transformer (ViT) that uses **attention-based token pruning** and **knowledge distillation** to cut inference cost while preserving accuracy — built and trained entirely on a CPU-only laptop (8GB RAM), no GPU required.

## Overview

Vision Transformers process every image patch ("token") through every layer, even tokens carrying little useful information (background, redundant regions). This project reduces that waste by:

1. **Token Pruning** — using attention scores from the CLS token to identify and drop low-importance tokens partway through the network, cutting compute in later layers.
2. **Knowledge Distillation** — training a smaller "student" model (with pruning) to match a larger "teacher" model's outputs, recovering the accuracy that pruning alone would sacrifice.

## Results

Trained from scratch on a CIFAR-10 subset (no pretrained weights), evaluated across multiple pruning ratios:

| Model | Keep Ratio | Accuracy | FLOPs (M) | Params (M) | Latency (ms) |
|---|---|---|---|---|---|
| Teacher (no pruning) | 1.0 | 35.8% | 154.79 | 2.385 | 16.08 |
| Student | 0.9 | 46.4% | 49.17 | 0.801 | 6.98 |
| **Student** | **0.7** | **52.2%** | **44.01** | **0.801** | **13.24** |
| Student | 0.5 | 44.9% | 39.25 | 0.801 | 7.17 |
| Student | 0.3 | 44.7% | 34.09 | 0.801 | 7.84 |

**Best configuration (keep_ratio = 0.7):**
- **71.6% FLOPs reduction** vs. teacher
- **2.27x faster inference**
- **5.8x fewer parameters**
- **+16.4% accuracy** vs. teacher (distillation + pruning act as effective regularization on the small training set)

See `results/plots/` for full accuracy-vs-FLOPs and latency comparison charts.

## Project Structure
efficient-vit-compression/
├── src/
│ ├── model.py # ViT architecture with optional mid-network pruning
│ ├── pruning.py # Attention-based token importance scoring + pruning
│ ├── distillation.py # Teacher-student setup + distillation loss
│ ├── train.py # Training loop (teacher + student ablation configs)
│ ├── evaluate.py # FLOPs / latency / memory / accuracy measurement
│ ├── plot_results.py # Generates ablation comparison charts
│ └── export_samples.py # Exports sample CIFAR-10 images for the demo UI
├── app.py # Streamlit demo — live teacher vs. student comparison
├── results/
│ ├── ablation_results.csv
│ └── plots/ # Accuracy vs. FLOPs, latency, params charts
├── sample_images/ # Real CIFAR-10 test images for the demo
└── requirements.txt




## How It Works

**Model**: A from-scratch Vision Transformer (patch embedding → transformer encoder blocks → classification head), sized small enough to train on CPU: student uses 6 layers / 128-dim embeddings, teacher uses 8 layers / 192-dim embeddings.

**Pruning**: After a configurable layer (default: layer 2), tokens are ranked by how much attention the CLS token pays to them, and only the top `keep_ratio` fraction survive into later layers — always preserving the CLS token itself.

**Distillation**: The student is trained on a weighted combination of standard cross-entropy loss (against true labels) and KL-divergence loss (against the teacher's softened output distribution), so it learns both the correct answer and *how confident/uncertain* the teacher was across all classes.

## Running Locally

```bash
# Set up environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Train teacher + student ablation configs
python src/train.py

# Evaluate all trained checkpoints
python src/evaluate.py

# Generate comparison charts
python src/plot_results.py

# Launch the interactive demo
streamlit run app.py
```

## Demo

The Streamlit app (`app.py`) lets you upload an image and see the teacher and student models predict side by side, with live inference timing and the ablation charts displayed below. Sample CIFAR-10 test images are included in `sample_images/` for quick testing.

## Notes on Scope

This project was built to demonstrate the compression *technique* and methodology end-to-end on constrained hardware  — trained on a subset of CIFAR-10 (5,000–10,000 images, 20 epochs) rather than the full dataset or a large-scale benchmark. Absolute accuracy is modest by design; the relative comparison across pruning ratios (the ablation trade-off curve) is the core contribution.

## Tech Stack

Python, PyTorch, Vision Transformers, Streamlit, Matplotlib, THOP (FLOPs profiling)
