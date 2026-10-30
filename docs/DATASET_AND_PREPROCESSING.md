# VoxGuard – Dataset Provenance and Audio Preprocessing Specification

This document details the audio data sources, partition integrity protocols, and mathematical signal preprocessing pipelines utilized in the **VoxGuard** AI Voice Deepfake Detection system.

---

## 1. Dataset Description: MLAAD-tiny

To bridge the gap between procedural software test fixtures and genuine real-speech detection without requiring the 30+ GB ASVspoof 2019 full archive, VoxGuard incorporates a controlled 2,000-sample subset of **MLAAD-tiny** (Multi-Language Audio Anti-spoofing Dataset).

- **Hugging Face Repository:** [mueller91/MLAAD-tiny](https://huggingface.co/datasets/mueller91/MLAAD-tiny)
- **Language Scope:** English (`en`) subset selected to eliminate cross-lingual phonetic confounders.
- **Licensing:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Primary Research Objective:** Train and benchmark fine-tuned transfer learning architectures against state-of-the-art neural speech synthesizers with explicit seen vs. unseen generator auditing.

---

## 2. Audio Sources and Class Distribution

The experimental corpus comprises exactly **2,000 audio recordings** in a balanced 1:1 binary distribution:

| Speech Class | Total Count | Acoustic Source / Generator | Characteristics |
|---|---:|---|---|
| **Genuine Human Speech (Bonafide)** | 1,000 | **LibriSpeech** / Public domain LibriVox audiobooks | Multi-speaker, natural prosody, acoustic room impulse response, variable pitch and accents. |
| **Synthetic Speech (Spoof / Deepfake)** | 1,000 | **8 Modern Neural Speech Synthesizers** | SOTA text-to-speech (TTS) and voice conversion (VC) architectures. |

### Generative Architectures in the Corpus

The 1,000 synthetic samples represent 8 distinct modern generative engines:

1. **Edge-TTS:** Microsoft Azure neural TTS engine (fluent prosody, high-frequency spectral smoothness).
2. **FishTTS:** Zero-shot cross-lingual autoregressive vocoder.
3. **Chatterbox:** Neural acoustic model with HiFi-GAN vocoding.
4. **DeepGram:** Commercial neural synthesis API.
5. **ZipVoice:** Diffusion-based fast speech synthesis.
6. **Cartesia.ai (Sonic-3):** Ultra-low-latency state-space / diffusion voice model (**Unseen Benchmark**).
7. **Mars5:** Autoregressive foundational speech generator (**Unseen Benchmark**).
8. **MatchaTTS:** Fast, non-autoregressive flow-matching acoustic model with HiFi-GAN (**Unseen Benchmark**).

---

## 3. Dataset Partitioning & Leakage Audit

The 2,000 audio files are partitioned strictly across Train, Development (validation), and Test sets to prevent data leakage and benchmark unseen generator generalization:

```
MLAAD-tiny Corpus (N = 2,000)
├── Training Partition (N = 1,400)
│   ├── Genuine Human: 700 samples (from disjoint audiobook sources)
│   └── Synthetic Spoof: 700 samples (5 Seen Generators: 140 samples each)
├── Development Partition (N = 200)
│   ├── Genuine Human: 100 samples (disjoint sources)
│   └── Synthetic Spoof: 100 samples (5 Seen Generators: 20 samples each)
└── Held-Out Test Partition (N = 400)
    ├── Genuine Human: 200 samples (150 evaluated in primary benchmark)
    └── Synthetic Spoof: 200 samples (150 evaluated in primary benchmark)
        ├── Seen Generators: 75 samples (Edge-TTS, FishTTS, Chatterbox, DeepGram, ZipVoice; 15 each)
        └── Unseen Generators: 75 samples (Cartesia Sonic-3, Mars5, MatchaTTS; 25 each)
```

### Partition Integrity Guarantees

1. **Zero Overlap:** Manifest audits confirm $\text{Files}(\text{Train}) \cap \text{Files}(\text{Dev}) \cap \text{Files}(\text{Test}) = \emptyset$.
2. **Source / Book Separation:** Genuine audio files are partitioned by audiobook title/chapter to minimize acoustic room and speaker leakage between splits.
3. **Generator Disjointness:** Three generative systems (`Cartesia.ai`, `Mars5`, and `MatchaTTS`) are **never seen** during training or development. They are reserved exclusively for evaluating zero-shot out-of-distribution generalization.
4. **Documented Metadata Limitation:** Canonical speaker IDs are not provided for the synthetic recordings in MLAAD-tiny; hence, speaker leakage cannot be mathematically audited for synthetic engines.

---

## 4. Audio Preprocessing Pipeline

VoxGuard implements a deterministic, multi-stage digital signal processing pipeline implemented in [`src/data/preprocessor.py`](file:///c:/VoxGuard/src/data/preprocessor.py).

### 4.1 Native Audio Decoding (Zero System Dependencies)
- Decodes `.wav`, `.flac`, and `.mp3` directly into single-precision 32-bit floating-point waveforms using `soundfile` (`libsndfile` v0.14.0+).
- Removes external dependencies on system-installed `ffmpeg.exe` for MP3 files.
- Rejects corrupt or truncated audio gracefully without silent interpretation.

### 4.2 Standard Resampling and Channel Standardization
- Converts all input signals to **$16,000\text{ Hz}$**, single-channel monophonic audio:
  $$x_{\text{mono}}[n] = \frac{1}{C}\sum_{c=1}^{C} x[n, c]$$
- Clamps amplitude to $[-1.0, 1.0]$.

### 4.3 Duration Handling: Dual-Mode Operation
- **Fixed-Duration Mode (Training & Procedural Fixtures):**
  - Standardizes audio to $4.0\text{ seconds}$ ($64,000\text{ samples}$ at $16\text{ kHz}$).
  - Audio $> 4.0\text{s}$ is sliced symmetrically from the energetic center.
  - Audio $< 4.0\text{s}$ is repeated cyclically until reaching $64,000$ samples.
- **Full-Duration Multi-Window Mode (Real-Speech Uploads in Demo):**
  - For user audio uploads, full audio waveforms are preserved (`normalize_duration = False`).
  - Utterances longer than $4.0\text{s}$ are segmented into sliding $4.0\text{s}$ windows with $2.0\text{s}$ hop stride.
  - Predictions are aggregated across all temporal windows via **median pooling**, preventing comical pitch inflections or isolated laughs from triggering harsh false positives.

### 4.4 Silence Trimming
- Trims leading and trailing silence below $30\text{ dB}$ relative to peak signal amplitude to eliminate silent channel artifacts from biasing classifier decisions.

---

## 5. Acoustic Feature Representations

VoxGuard supports three distinct acoustic representations tailored to specific architectural paradigms:

### A. 128-bin Log-Mel Spectrogram (Fine-Tuned Real-Speech Detector)
- **Target Model:** Pretrained ResNet-18 + BiGRU + Attention detector.
- **STFT Parameters:**
  - Sample Rate: $16,000\text{ Hz}$
  - FFT Size ($N_{\text{fft}}$): $1,024$
  - Window Length: $400$ samples ($25.0\text{ ms}$)
  - Hop Length: $160$ samples ($10.0\text{ ms}$)
  - Window Function: Periodic Hann window
- **Mel Filterbank:** $128$ triangular Mel filter bands spanning $0\text{ Hz}$ to $8,000\text{ Hz}$ (Nyquist).
- **Logarithmic Compression:**
  $$S_{\text{log}}[f, t] = \log_{10}\left(S_{\text{mel}}[f, t] + 10^{-6}\right)$$
- **Output Tensor:** $(1, 1, 128, T)$, where $T \approx 401$ for a standard $4.0\text{s}$ window.

### B. 64-bin Log-Mel Spectrogram (Procedural Software-Test Fixture CNN)
- **Target Model:** 2D Spectrogram-CNN (`src/models/spec_cnn.py`).
- **Mel Filterbank:** $64$ triangular filter bands ($20\text{ Hz}$ to $8,000\text{ Hz}$).
- **Output Tensor:** $(1, 1, 64, 401)$.

### C. 120-dimensional Summary MFCC Statistics (Baseline Classifiers)
- **Target Models:** Logistic Regression & Balanced Random Forest.
- **Coefficients:** $20$ static Mel-Frequency Cepstral Coefficients.
- **Derivatives:** First ($\Delta$) and second ($\Delta^2$) temporal velocity and acceleration derivatives ($20 + 20 + 20 = 60$ frames).
- **Summary Statistics:** Temporal mean ($\mu$) and standard deviation ($\sigma$) across all 60 dimensions:
  $$\mathbf{f}_{\text{MFCC}} = \left[\mu_1, \dots, \mu_{60}, \sigma_1, \dots, \sigma_{60}\right] \in \mathbb{R}^{120}$$
- **Normalization:** `StandardScaler` fitted strictly on training data; dev and test sets transformed using frozen parameters.

---

## 6. Biological Vocal Periodicity Verification ($R_{\text{max}}$)

To eliminate false-positive classifications on intentional human vocal modulations (e.g., mimicry, comedy skits, impressions), VoxGuard computes the normalized pitch autocorrelation of the waveform:

$$R(\tau) = \frac{\sum_{n} x[n] x[n+\tau]}{\sqrt{\sum_{n} x[n]^2 \sum_{n} x[n+\tau]^2}}$$

- **Lag Range ($\tau$):** $[32, 320]$ samples at $16\text{ kHz}$, corresponding to the human glottal fundamental frequency range ($50\text{ Hz} - 500\text{ Hz}$).
- **Metric ($R_{\text{max}}$):** Peak correlation within the human pitch range.
- **Forensic Rule:** If an utterance falls in the decision boundary uncertainty band ($0.45 \le p \le 0.62$) but possesses $R_{\text{max}} \ge 0.35$, VoxGuard verifies physical vocal fold vibration, classifies the utterance as **Human Speech (Vocal Mimicry Detected)**, and scales the confidence to bonafide baseline.

