# Zeek-Context-Transformer
Self-supervised contextual anomaly detection pipeline for network flows using a custom Transformer architecture.
# Zeek Context Transformer: Self-Supervised Anomaly Detection for Network Flows

This repository contains the core implementation of a specialized **Self-Supervised Masked Autoencoder** designed to process network security telemetry. This architecture serves as the underlying contextual analysis layer for an end-to-end network intrusion detection framework utilizing **Zeek** log data.

## 🚀 Concept & Methodology
Traditional intrusion detection systems rely on static signatures or reactive anomaly thresholds. This model introduces a proactive approach by adapting the concept of **Masked Language Modeling (MLM)**—commonly used in NLP Transformers—to continuous, high-dimensional network flow sequences.

1. **Contextual Encoding:** Network connection logs (e.g., connection status, byte sizes, duration, packet counts extracted via Zeek) are segmented into sequential windows.
2. **Self-Supervised Masking:** During the pre-training phase, the `mask_flows` function randomly hides a percentage of network flows within each sequence window.
3. **Reconstruction & Anomaly Scoring:** The Deep Transformer Encoder is trained to reconstruct the hidden elements by learning the underlying structural and temporal boundaries of normal network behavior. 

When exposed to a **zero-day attack** or unexpected scanning behavior, the model fails to accurately reconstruct the anomalous sequence, leading to a sharp spike in the calculated reconstruction error (`anomaly_score`).

## 🛠️ Model Architecture & Pipeline
- **Positional Embedding:** A trainable embedding layer to capture the temporal order of network events.
- **Custom Transformer Encoder Layer:** Built natively using TensorFlow/Keras multi-head attention blocks and feed-forward networks for robustness.
- **Dynamic Masking System:** Handled natively to compute losses only on hidden states, bypassing static graph dependencies.

## 💻 Quick Start & Usage

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- NumPy

### Running the Pipeline
You can run the full end-to-end pipeline (data window generation, masking pre-training, and inference testing) by executing:

```bash
python main.py
