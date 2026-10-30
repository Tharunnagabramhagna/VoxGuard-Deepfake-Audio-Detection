---
language: en
license: apache-2.0
tags:
  - audio-classification
  - deepfake-detection
  - voice-detection
  - wav2vec2
  - transfer-learning
  - pytorch
  - speech-processing
  - binary-classification
metrics:
  - accuracy
  - precision
  - recall
  - f1
  - auc-roc
datasets:
  - asvspoof-2021
  - wavefake-test
  - audio-deepfake
  - fake-real-audio
  - deepfake-audio
  - combined-real-voices
  - scenefake
  - gender-balanced-audio-deepfake
  - synthetic-speech-commands
---

# Deepfake Voice Detector - SOTA Transfer Learning

Repository: https://huggingface.co/koyelog/deepfake-voice-detector-sota  
Model card last updated: 2025-10-31

This repository contains a binary audio classification model trained to distinguish real (bonafide) human voice samples from fake (synthetic / deepfake) voice samples. The model leverages transfer learning from facebook/wav2vec2-base and a lightweight temporal classifier (BiGRU + Multi-Head Attention) on top.

## Model Details

- Model name: Deepfake Voice Detector - SOTA Transfer Learning  
- HF repo: koyelog/deepfake-voice-detector-sota  
- Task: Audio Classification (Binary — Real vs Fake)  
- Base model: facebook/wav2vec2-base (feature extractor)  
- Architecture:
  - Wav2Vec2 encoder (facebook/wav2vec2-base) — pretrained speech feature extractor; CNN layers frozen
  - Bidirectional GRU: 2 layers, 256 hidden units per direction (512 total)
  - Multi-Head Attention: 8 heads, 512-dimensional embeddings
  - Classification head:
    - Linear(512 → 512) + ReLU + BatchNorm + Dropout(0.4)
    - Linear(512 → 128) + ReLU + BatchNorm + Dropout(0.3)
    - Linear(128 → 1) + Sigmoid
- Framework: PyTorch + Transformers
- Total parameters: ~98.5M
- Trainable parameters: ~98.5M
- Input: 4-second audio clip at 16 kHz (single-channel)
- Output: single probability (0..1) representing likelihood of "fake". Default threshold: 0.5 (0 = Real, 1 = Fake)
- License: Apache-2.0

## Training Procedure

- Training data: 822,166 audio samples aggregated from 19 datasets (listed below)
  - Real/Bonafide: 387,422 samples (47.1%)
  - Fake/Deepfake/Synthetic: 434,744 samples (52.9%)
- Dataset sources (combined): ASVspoof 2021, WaveFake, Audio-Deepfake, Fake-Real-Audio, Deepfake-Audio, Combined-Real-Voices, Scenefake, Gender-Balanced-Audio-Deepfake, Synthetic-Speech-Commands, and 10+ other Kaggle/academic datasets
- Data preprocessing:
  - Resample to 16 kHz
  - Fixed-length segments: 4 seconds (pad/truncate as required)
  - Feature extraction: raw audio → wav2vec2 feature frames
  - Data balancing & augmentation: dataset composition described above; standard augmentations (noise, speed perturbation) used where applicable
- Train / Val split:
  - Training: 657,732 samples (80%)
  - Validation: 164,434 samples (20%)
- Optimization:
  - Optimizer: AdamW
  - Learning rate: 5e-5
  - Weight decay: 0.01
  - Batch size: 24
  - Gradient accumulation: 2 (effective batch size 48)
  - Epochs: 20
  - Scheduler: Cosine Annealing with Warm Restarts (T_0=5, T_mult=2)
  - Loss: Binary Cross-Entropy (BCE)
  - Mixed precision: supported where hardware permits
- Hardware: Tesla P100-PCIE-16GB
- Training time: ~16 hours (single GPU as reported)
- Random seed and reproducibility: users should set deterministic seeds for fully reproducible runs (not included in this artifact)

## Evaluation Results

The reported validation metrics (expected ranges from final evaluation) are:

- Validation accuracy: 95%–97%
- Precision: ~0.95
- Recall: ~0.94
- F1-score: ~0.94
- AUC-ROC: ~0.96

Notes:
- These values represent evaluation on the combined held-out validation split described above. Performance will vary by dataset, language, recording conditions, and unseen manipulation techniques.
- Reported metrics are aggregated and averaged across the validation partition. Per-dataset metrics (e.g., ASVspoof vs WaveFake) will differ and are not included in this artifact.

## How to Use

Supported input: 4-second audio clip sampled at 16 kHz. Longer or shorter clips should be truncated or padded to 4 seconds before inference.

Example (pseudocode / minimal PyTorch + Transformers usage):

```python
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor
# model = load your checkpoint wrapped with the BiGRU+Attention classifier

# 1) Load audio and resample to 16k
waveform, sr = torchaudio.load("example.wav")
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

# 2) Ensure 4 seconds length (pad or truncate)
target_len = 4 * 16000
if waveform.shape[1] < target_len:
    pad = target_len - waveform.shape[1]
    waveform = torch.nn.functional.pad(waveform, (0, pad))
else:
    waveform = waveform[:, :target_len]

# 3) Feature extraction (wav2vec2)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
input_values = feature_extractor(waveform.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt").input_values

# 4) Forward pass through model
model.eval()
with torch.no_grad():
    logits = model(input_values)          # model should return a scalar logit per sample
    prob = torch.sigmoid(logits).item()   # probability of 'fake' class

prediction = 1 if prob >= 0.5 else 0
confidence = prob if prediction == 1 else 1 - prob
```

Model outputs:
- logits / raw score (float)
- probability (sigmoid(logit)): float in [0,1]
- final label: 0 = Real (bonafide), 1 = Fake (deepfake/synthetic)
- confidence: probability of the predicted class

## Labeling Strategy

- Label 0 — Real / Bonafide:
  - Human voice recordings from authentic sources (keywords in metadata/filenames: bonafide, real, genuine, human, authentic, original)
- Label 1 — Fake / Deepfake:
  - AI-generated, synthetic, or manipulated audio produced by text-to-speech, voice conversion, or other spoofing methods (keywords: spoof, fake, deepfake, synthetic, generated, ai)

Labels were assigned by dataset provider metadata and cross-checked using dataset documentation. Users applying new datasets should ensure consistent labeling and metadata mapping.

## Limitations and Biases

- Clip length sensitivity: The model is optimized for 4-second clips; performance on markedly shorter/longer clips may degrade.
- Language & accent coverage: Although trained on many datasets and multi-language samples, underrepresented languages/accents in the training corpora can cause degraded performance.
- Dataset composition: Slight skew towards fake samples (52.9% fake vs 47.1% real) — may increase false positives in certain deployment scenarios.
- Novel attacks: Not evaluated on zero-shot or post-2025 deepfake generation techniques; performance against new generator families is unknown.
- Environmental factors: Recording quality, background noise, channel effects, and codecs may affect predictions.
- Ethical risk: Incorrect or automated use of the model can cause reputational or legal harm. Model outputs should not be used as sole evidence.

## Ethical Considerations

- This model is intended as an assistive tool for verification and detection workflows. Human oversight is essential for any high-stakes decisions.
- Avoid using the model to definitively accuse individuals of wrongdoing without corroborating evidence.
- Respect privacy and legal restrictions when processing audio data.
- Be transparent about limitations, false positive/negative rates, and the potential for demographic biases.

## Hardware & Inference Requirements

- Recommended: GPU with CUDA support for fast inference (e.g., NVIDIA GPUs). CPU inference possible but slower.
- Approximate memory: ~2 GB GPU VRAM for single-sample inference (depends on implementation).
- Batch inference recommended for throughput.

## Caveats & Reproducibility

- This model card documents a trained model artifact. If you require retraining or further experiments, use the architecture specification above and provide deterministic seeds and full data provenance to replicate results.
- Exact training scripts, hyperparameter sweep logs, and raw dataset bundles are not included in this model artifact. Users should exercise caution when assuming identical performance in other contexts.

## Citation

If you use this model, please cite:

@misc{deepfake-voice-detector-sota,
  author = {koyelog},
  title = {Deepfake Voice Detector - Transfer Learning with Wav2Vec2},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/koyelog/deepfake-voice-detector-sota}
}

## Contact

Model owner: koyelog  
Model hub: https://huggingface.co/koyelog/deepfake-voice-detector-sota

```