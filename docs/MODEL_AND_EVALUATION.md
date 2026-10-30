# VoxGuard – Model Architectures, Fine-Tuning & Evaluation Benchmarks

This document records the architectural details, transfer learning and fine-tuning protocols, checkpoint provenance, and verified experimental benchmarks for the **VoxGuard** AI Voice Deepfake Detection system.

---

## 1. Model Architectures Implemented

VoxGuard integrates two tiers of models:
1. **Procedural Software Test Harness Models:** Designed for CPU test verification, mathematical calibration, and CI/CD testing.
2. **Deep Neural Real-Speech Detector:** Reconstructed and fine-tuned for authentic human vs. AI-synthesized speech discrimination.

### 1.1 Procedural Software-Test Fixture Models

| Model Architecture | Input Features | Parameters | Training Mechanism | Role |
|---|---|---:|---|---|
| **Logistic Regression** | 120-dim MFCC summary ($\mu, \sigma, \Delta, \Delta^2$) | 121 | $L_2$-regularized convex optimization | Classical linear acoustic baseline |
| **Random Forest** | 120-dim MFCC summary | ~15,000 | 150 balanced decision trees, max depth 18 | Non-linear ensemble baseline |
| **Spectrogram-CNN (2D CNN)** | 64-bin Log-Mel Spectrogram ($64 \times 401$) | 1,273,473 | 4 Conv blocks (Conv2d, BatchNorm, ReLU, MaxPool, Dropout) + Linear head | Deep convolutional spatial baseline |

### 1.2 Fine-Tuned Real-Speech Deepfake Detector

- **Base Provenance:** Pretrained transfer learning checkpoint derived from `koyelog/deepfake-voice-detector-sota` (Apache-2.0 license).
- **Composite Architecture:** ResNet-18 Convolutional Backbone + Bidirectional GRU + Multi-Head Self-Attention + MLP Classification Head.
- **Total Parameters:** $6,129,089$ (23.38 MB uncompressed float32).

```
Input Audio Waveform (16 kHz, Mono)
      │
      ▼
128-bin Log-Mel Spectrogram (Shape: [B, 1, 128, 401])
      │
      ▼  [FROZEN BACKBONE: 2,776,512 params | Eval Mode | BatchNorm Frozen]
ResNet-18 Feature Extractor (conv1, bn1, layer1, layer2, layer3)
      │  Output Shape: [B, 256, 16, 51] -> Permuted & Flattened -> [B, 51, 4096]
      ▼
Projection Layer (Linear: 4096 -> 512)
      │
      ▼  [TRAINABLE HEAD: 3,352,577 params (54.7%)]
Bidirectional GRU (2 layers, hidden_size=256, bidirectional=True -> 512 dim)
      │
      ▼
Multi-Head Self-Attention (8 heads, embed_dim=512)
      │  Temporal Pooling (Global Average Pooling across frames)
      ▼
Classifier MLP Head:
  Linear(512 -> 512) -> BatchNorm1d -> ReLU -> Dropout(0.4)
  Linear(512 -> 128) -> BatchNorm1d -> ReLU -> Dropout(0.3)
  Linear(128 -> 1) -> Sigmoid Posterior Probability
```

---

## 2. Fine-Tuning Protocol (Phase 2B)

Fine-tuning was conducted on the audited MLAAD-tiny 1,400-sample training partition with rigorous parameter freezing safeguards:

1. **Backbone Freezing:**
   - All parameters in `conv1`, `bn1`, `layer1`, `layer2`, and `layer3` ($2,776,512$ weights/biases) had `requires_grad = False`.
   - All 15 BatchNorm layers inside the ResNet backbone were forced into evaluation mode (`m.eval()`) during training, preserving `running_mean`, `running_var`, and `num_batches_tracked` bit-for-bit.
2. **Trainable Subsystems:**
   - Projection linear layer, 2-layer BiGRU, 8-head Attention mechanism, and MLP classifier ($3,352,577$ parameters; $54.7\%$ of network).
3. **Hyperparameters:**
   - **Batch Size:** $2$ (optimized for CPU/memory efficiency).
   - **Optimizer:** AdamW ($\text{lr} = 5 \times 10^{-5}$, weight decay $= 0.01$).
   - **Loss Function:** `BCEWithLogitsLoss` (balanced 1:1 classes).
   - **Random Seed:** $42$ (fixed and recorded).
   - **Max Epochs:** $12$ with early stopping based on development-set loss (patience $= 3$ epochs).
4. **Model Selection:**
   - Best checkpoint was selected strictly based on minimum development loss (Epoch 2: $\text{Dev Loss} = 0.5960$), saving `checkpoints/real_speech/voxguard_mlaad_finetuned/best_model.pt`.

---

## 3. Real-Speech Evaluation Benchmarks (MLAAD-tiny Held-Out Test Set)

Evaluated on the held-out test partition ($N=300$: $150$ genuine human LibriSpeech samples, $150$ synthetic samples across 8 generators).

### 3.1 Overall Performance Comparison

| Metric | Pretrained Baseline ($\theta = 0.50$) | Fine-Tuned Model ($\theta = 0.50$) | Baseline (Dev Calibrated $\theta = 0.0758$) | Fine-Tuned (Dev Calibrated $\theta = 0.2777$) |
|---|---:|---:|---:|---:|
| **Accuracy** | 58.67% | **66.67%** (+8.0%) | 63.33% | 63.00% |
| **Precision** | **93.33%** | 86.76% | 62.35% | 61.68% |
| **Recall (Spoof Detection)** | 18.67% | **39.33%** (+110% rel) | 67.33% | **68.67%** |
| **F1-Score** | 0.3111 | **0.5413** (+74% rel) | 0.6474 | **0.6498** |
| **ROC-AUC** | 0.6958 | 0.6948 | 0.6958 | 0.6948 |
| **Equal Error Rate (EER)** | 35.33% | 36.67% | 35.33% | 36.67% |
| **False Positive Rate (FPR)** | **1.33%** (2/150) | 6.00% (9/150) | 40.67% (61/150) | 42.67% (64/150) |
| **False Negative Rate (FNR)** | 81.33% (122/150) | **60.67%** (91/150) | 32.67% (49/150) | **31.33%** (47/150) |
| **Confusion Matrix** | TN=148, FP=2, FN=122, TP=28 | **TN=141, FP=9, FN=91, TP=59** | TN=89, FP=61, FN=49, TP=101 | TN=86, FP=64, FN=47, TP=103 |

> [!NOTE]
> Under default threshold $\theta = 0.50$, fine-tuning more than doubled true positive detections ($28 \to 59$ detected deepfakes), increasing F1-score from $0.3111$ to $0.5413$ while maintaining a low false-positive rate on human speech ($6.0\%$).

---

### 3.2 Breakdown by Seen vs. Unseen Speech Generators

| Generator Partition | Evaluation Subset | Baseline Accuracy ($\theta = 0.50$) | Fine-Tuned Accuracy ($\theta = 0.50$) | Baseline Accuracy (Dev Calibrated) | Fine-Tuned Accuracy (Dev Calibrated) |
|---|---|---:|---:|---:|---:|
| **Human Speech (Bonafide)** | 150 LibriSpeech samples | **98.67%** (148/150) | 94.00% (141/150) | 59.33% (89/150) | 57.33% (86/150) |
| **Seen Generators** | 75 samples (5 systems) | 17.33% (13/75) | **36.00%** (27/75) | 64.00% (48/75) | **65.33%** (49/75) |
| **Unseen Generators** | 75 samples (3 systems) | 20.00% (15/75) | **42.67%** (32/75) | 70.67% (53/75) | **72.00%** (54/75) |

---

### 3.3 Detailed Per-Generator Detection Accuracy

| Generator / System | Status | Sample Count | Baseline Acc ($\theta = 0.50$) | Fine-Tuned Acc ($\theta = 0.50$) | Fine-Tuned Acc (Calibrated $\theta = 0.278$) | Mean Posterior Prob ($\hat{p}$) |
|---|---|---:|---:|---:|---:|---:|
| **Human Speech** | Bonafide | 150 | **98.7%** (148) | 94.0% (141) | 57.3% (86) | 0.287 |
| **Edge-TTS** | Seen | 15 | 46.7% (7) | **73.3%** (11) | **100.0%** (15) | 0.662 |
| **FishTTS** | Seen | 15 | 0.0% (0) | 0.0% (0) | 13.3% (2) | 0.180 |
| **Chatterbox** | Seen | 15 | 0.0% (0) | **13.3%** (2) | 53.3% (8) | 0.321 |
| **DeepGram** | Seen | 15 | 26.7% (4) | **53.3%** (8) | **93.3%** (14) | 0.566 |
| **ZipVoice** | Seen | 15 | 13.3% (2) | **40.0%** (6) | 66.7% (10) | 0.462 |
| **Cartesia.ai (Sonic-3)** | **Unseen** | 25 | 20.0% (5) | **48.0%** (12) | **72.0%** (18) | 0.463 |
| **Mars5** | **Unseen** | 25 | 0.0% (0) | **12.0%** (3) | 52.0% (13) | 0.305 |
| **MatchaTTS** | **Unseen** | 25 | 40.0% (10) | **68.0%** (17) | **92.0%** (23) | 0.644 |

---

## 4. Software-Test Harness Procedural Fixture Benchmarks

For deterministic software validation on procedural fixtures ($N=30$ mini and $N=50$ CD-ADD synthetic fixtures):

| Model | Dev Threshold | Eval Acc (Mini N=30) | F1-Score | ROC-AUC | CD-ADD Test Acc (N=50) | CD-ADD EER (%) | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| **Logistic Regression** | $\theta = 0.3903$ | 26.67% | 0.3889 | 0.1150 | 22.00% | 72.00% | Verified Fixture |
| **Random Forest** | $\theta = 0.5546$ | 16.67% | 0.1935 | 0.1450 | 32.00% | 72.00% | Verified Fixture |
| **Spectrogram-CNN** | $\theta = 0.4604$ | 30.00% | 0.3226 | 0.2850 | 24.00% | 72.00% | Verified Fixture |

---

## 5. Checkpoint Provenance & Availability

| Checkpoint Path | Architecture | Size | In Git? | Download / Source |
|---|---|---:|---|---|
| `checkpoints/real_speech/voxguard_mlaad_finetuned/best_model.pt` | ResNet-18 + BiGRU + Attention | 49.04 MB | **YES** | Tracked in Git; fine-tuned on MLAAD-tiny |
| `checkpoints/real_speech/voxguard_mlaad_finetuned/provenance.json` | JSON Metadata | 1.8 KB | **YES** | Tracked in Git; records hash and parameters |
| `checkpoints/real_speech/voxguard_mlaad_finetuned/training_history.json` | JSON Metrics | 2.1 KB | **YES** | Tracked in Git; epoch-by-epoch loss log |
| `checkpoints/real_speech/wav2vec2_deepfake_detector/pytorch_model.pth` | ResNet-18 + BiGRU + Attention | 70.27 MB | **Preserved Locally** | Download via `python scripts/download_pretrained_model.py` from Hugging Face (`koyelog/deepfake-voice-detector-sota`) |
| `checkpoints/fixture/best_spec_cnn.pt` | 2D Spectrogram-CNN | 4.85 MB | **YES** | Tracked in Git; procedural test fixture |
| `checkpoints/fixture/logistic_regression.joblib` | Scikit-Learn Pipeline | 10.2 KB | **YES** | Tracked in Git |
| `checkpoints/fixture/random_forest.joblib` | Scikit-Learn Ensemble | 161.4 KB | **YES** | Tracked in Git |

