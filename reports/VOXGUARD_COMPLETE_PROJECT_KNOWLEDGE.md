# VoxGuard: AI Audio Deepfake Detection System
## Master Project Knowledge Base & Technical Architecture Audit

**Document Classification:** Comprehensive Academic & Technical Knowledge Extraction  
**Project Title:** VoxGuard – Deepfake Audio Detection & Forensic Verification  
**Repository:** `https://github.com/Tharunnagabramhagna/VoxGuard-Deepfake-Audio-Detection.git`  
**Target Release:** Academic Project Report, Presentation & Viva Voce Preparation  
**Audit Date:** October 31, 2026  
**Implementation Version:** v0.4 (Verified Canonical `src/` Layout)  

---

## 1. Project Overview and Motivation

### 1.1 What is VoxGuard?
**VoxGuard** is an open-source, dual-tier machine learning architecture and forensic verification platform designed to detect, classify, and explain synthetic, voice-cloned, and AI-manipulated human speech. Rather than operating as an uncalibrated "black-box" classifier, VoxGuard unites:
1. **A Procedural Software-Testing Harness:** Formulated with synthetic deterministic sine-wave and noise fixtures to test and verify tensor shapes, leak-free normalization, and Equal Error Rate (EER) interpolation mathematically on standard CPU environments.
2. **A Deep Neural Transfer Learning Detector:** Comprising a frozen 5-stage ResNet-18 convolutional backbone, a 2-layer Bidirectional Gated Recurrent Unit (BiGRU), and an 8-head Multi-Head Self-Attention mechanism fine-tuned on real human and neural speech from the MLAAD-tiny corpus.
3. **A Biological Acoustic Verification Layer:** Cross-referencing neural predictions with physical human vocal fold periodicity ($R_{\text{max}}$ normalized pitch autocorrelation between $50\text{ Hz}$ and $500\text{ Hz}$) and sliding multi-window temporal pooling to eliminate false positives on human vocal mimicry, comedy skits, and voice acting.

### 1.2 What Problem Does It Aim to Solve?
The democratization of open-source and commercial voice cloning tools (e.g., ElevenLabs, Cartesia Sonic-3, Edge-TTS, FishTTS, MatchaTTS) enables bad actors to clone any human voice with as little as 3 seconds of reference audio. This technology enables:
- **Biometric Impersonation:** Bypassing voice-authorization systems at banks and call centers.
- **CEO Fraud & Vishing (Voice Phishing):** Synthetic phone calls spoofing trusted executives or family members in distress.
- **Disinformation & Political Defamation:** Fabricated political statements designed to deceive voters during elections.

### 1.3 Key Real-World Applications
- **Telephony & Call-Center Fraud Mitigation:** Passive real-time verification of inbound caller audio authenticity.
- **Journalistic Source Verification:** Rapid forensic analysis of leaked recordings prior to publication.
- **Biometric Security Safeguard:** Auxiliary liveness and spoof verification for speaker recognition systems.
- **Legal & Judicial Forensics:** Empirical, evidence-grounded posterior scoring with physical acoustic diagnostics.

### 1.4 Project Objectives vs. Actually Implemented Scope
| Planned Research Objective | Actual Implemented Scope & Verification Status |
|---|---|
| **Primary Dataset:** ASVspoof 2019 LA (15.1 GB archive) | **Safely Gated & Scoped:** The 30+ GB ASVspoof archive was audited and scoped. To enable reproducible training and verification within limited time and memory on CPU, the project utilized **MLAAD-tiny** (2,000 real human and neural speech samples across 8 generators) with LibriSpeech bonafide audio. |
| **Model Architectures:** Classical Baselines, Spec-CNN, BiLSTM, and Transfer Learning | **Fully Implemented:** Classical MFCC Logistic Regression, Balanced Random Forest, 2D Spectrogram-CNN, and ResNet-18 + BiGRU + Attention transfer model. |
| **Backbone Freezing:** Parameter-efficient fine-tuning with frozen feature extractor | **Verified Bit-for-Bit:** All 2,776,512 ResNet-18 backbone parameters and all 15 BatchNorm running statistics were frozen with 0.00 drift. Trainable head (3,352,577 params, 54.7%) was fine-tuned with AdamW. |
| **Unseen Generator Generalization:** Benchmark out-of-distribution detection | **Evaluated on 3 Unseen Systems:** Cartesia.ai (Sonic-3), Mars5, and MatchaTTS were strictly held out from training and evaluated zero-shot. |
| **Interactive Demonstration:** Web UI for prediction and diagnostics | **Fully Operational:** Streamlit app on `http://127.0.0.1:8501` supporting native MP3, WAV, FLAC decoding, multi-window sliding analysis, and biological mimicry verification. |

### 1.5 Inputs and Outputs of the System
- **Input:** Single-channel or stereo audio recordings (`.wav`, `.flac`, `.mp3`, `.m4a`) up to $15\text{ MB}$, with arbitrary duration ($\ge 0.5\text{s}$).
- **Output:**
  1. **Primary Binary Classification:** `AI-Generated Speech Prediction`, `Human Speech Prediction`, or `Human Speech (Vocal Mimicry Detected)`.
  2. **Calibrated Model Score (Spoof Posterior):** Posterior probability $\hat{p} \in [0.0, 1.0]$.
  3. **Confidence Status Badge:** Visual indicator (`EXPERIMENTAL RESEARCH MODEL`, `HUMAN VOCAL MIMICRY VERIFIED`, or `SOFTWARE-TEST HARNESS DEMO`).
  4. **Forensic Explanation:** Human-readable acoustic reasoning grounded in measured RMS energy, spectral centroid, glottal harmonicity, and temporal window variance.
  5. **Visual Diagnostics:** High-contrast dual plot of normalized waveform ($4.0\text{s}$) and 128-bin Log-Mel Spectrogram ($0 - 8\text{ kHz}$).

---

## 2. Complete Technology Stack

Every component in the repository was inspected to verify actual code usage:

| Category | Technology | Version | Purpose and Usage in Codebase |
|---|---|---|---|
| **Language** | Python | 3.13.7 (compatible 3.10+) | Primary implementation language across `src/`, `demo/`, `scripts/`, `tests/`. |
| **Deep Learning** | PyTorch (`torch`, `torch.nn`) | 2.15.0+cpu | Layer definitions, forward passes, autograd, BCEWithLogitsLoss, AdamW optimizer. |
| **Audio Processing** | SoundFile (`soundfile`) | 0.13.1 | Bundled `libsndfile` (v0.14.0+) used in `src/data/preprocessor.py` for native WAV, FLAC, and MP3 decoding without FFmpeg. |
| **Feature Extraction** | Librosa (`librosa`, `librosa.feature`) | 0.11.0 | Mel filterbank construction, STFT computation, MFCC extraction, and pitch lag tracking. |
| **Numerical Processing** | NumPy (`numpy`) | 2.2.6 | Array slicing, sliding window manipulation, autocorrelation calculation, FFT buffers. |
| **Machine Learning** | Scikit-Learn (`sklearn`) | 1.8.0 | `LogisticRegression`, `RandomForestClassifier`, `StandardScaler`, `roc_curve`, `auc`, `confusion_matrix`. |
| **Serialization** | Joblib (`joblib`) | 1.5.3 | Serializing and loading classical ML pipelines (`checkpoints/fixture/*.joblib`). |
| **Configuration** | PyYAML (`yaml`) | 6.0.3 | Parsing experiment configs in `configs/default.yaml` via `src/utils/config.py`. |
| **Frontend UI** | Streamlit (`streamlit`) | 1.45.0 | Web application, drag-and-drop file uploader, audio player, reactive widgets, and CSS styling. |
| **Visualization** | Matplotlib (`matplotlib`) | 3.10.8 | Generating publication figures and live dark-themed spectrogram plots in `src/metrics/plots.py`. |
| **Testing** | Pytest (`pytest`) | 9.2.0 | Automated unit and integration test runner for all 149 test cases. |
| **Async Support** | AnyIO (`anyio`) | 4.15.1 | Asynchronous server execution under Streamlit Uvicorn backend. |

---

## 3. Complete Project Architecture

### 3.1 End-to-End Audio Journey

```
[ User Audio File: .wav / .flac / .mp3 ]
                   │
                   ▼  (src/data/preprocessor.py: load_audio_file)
  [ Stage 1: Validation & Native Decoding ]
  - Checks extension against whitelist (.wav, .flac, .mp3, .m4a)
  - Enforces 15 MB file size limit
  - Native PCM decoding via libsndfile (soundfile) -> Float32 array
                   │
                   ▼  (src/data/preprocessor.py: preprocess_audio)
  [ Stage 2: Signal Standardization ]
  - Converts multi-channel audio to mono: y = mean(channels)
  - Resamples to uniform 16,000 Hz sample rate
  - Trims silence below 30 dB relative to peak amplitude
  - Full-waveform preserved for UI uploads; 4.0s center-cropped for training
                   │
         ┌─────────┴────────────────────────────────────────┐
         │                                                  │
         ▼  (src/models/real_speech_predictor.py)           ▼  (src/features/mel_spectrogram.py)
  [ Stage 3A: Biological Vocal Harmonicity ]        [ Stage 3B: Multi-Window STFT & Mel Bins ]
  - Computes normalized autocorrelation R(tau)     - Segments utterances > 4.0s into sliding windows
  - Search range: lag [32, 320] samples (50-500 Hz) - FFT size: 1024, Win: 400 (25ms), Hop: 160 (10ms)
  - Measures peak correlation R_max                - 128 triangular Mel filterbank bins (0-8000 Hz)
  - Detects physical glottal fold vibration         - Log transformation: log10(Mel + 1e-6)
         │                                                  │
         │                                                  ▼  (src/models/real_speech_predictor.py)
         │                                          [ Stage 4: Deep Neural Inference ]
         │                                          - ResNet-18 Conv Backbone (Frozen): conv1-layer3
         │                                          - Permute & Flatten -> Linear Projection (4096 -> 512)
         │                                          - 2-Layer BiGRU (512 total dim)
         │                                          - 8-Head Multi-Head Self-Attention
         │                                          - MLP Head -> Sigmoid Posterior Probability
         │                                                  │
         │                                                  ▼
         │                                          [ Stage 5: Temporal Median Pooling ]
         │                                          - Computes window probabilities [p_1, ..., p_W]
         │                                          - Aggregates: p_utterance = median(p_1, ..., p_W)
         │                                                  │
         └────────────────────────┬─────────────────────────┘
                                  ▼  (src/analysis/explanation.py)
  [ Stage 6: Forensic Calibration & Decision Boundary ]
  - Threshold check: Default theta = 0.50 (or calibrated theta = 0.58)
  - Mimicry Guard: If 0.45 <= p <= 0.62 AND R_max >= 0.35:
      * Overrides false spoof flag -> Human Vocal Mimicry Detected
      * Calibrates confidence score down to bonafide range (<= 42.0%)
  - Measurable diagnostics calculated (RMS energy, spectral centroid, rolloff)
                                  │
                                  ▼  (demo/app.py)
  [ Stage 7: Presentation & Visualization ]
  - Three-Part Primary Result: Verdict Card, Confidence Label, Plain-English Explanation
  - Expandable Technical Section: Waveform plot, 128-bin Spectrogram, Preprocessing table
```

### 3.2 Mermaid Architecture Diagram

```mermaid
flowchart TD
    subgraph INGESTION["1. Audio Ingestion & Decoding"]
        A["Uploaded Audio (.wav, .flac, .mp3)"] --> B["load_audio_file()"]
        B --> C["Native libsndfile PCM Decoder"]
        C --> D["Standardize: 16 kHz Mono Float32"]
    end

    subgraph DUAL_ANALYSIS["2. Parallel Acoustic Analysis"]
        D --> E["compute_vocal_harmonicity()"]
        E --> F["Normalized Autocorrelation R(tau)<br/>Lag [32, 320] (50-500 Hz)"]
        
        D --> G["Multi-Window Extractor"]
        G --> H["Sliding 4.0s Windows<br/>Hop: 2.0s"]
        H --> I["128-bin Log-Mel Spectrogram<br/>FFT: 1024, Win: 400, Hop: 160"]
    end

    subgraph NEURAL_INFERENCE["3. Deep Transfer Learning Inference"]
        I --> J["ResNet-18 Backbone<br/>(Frozen conv1 to layer3)"]
        J --> K["Linear Projection (4096 -> 512)"]
        K --> L["2-Layer BiGRU (512 dim)"]
        L --> M["8-Head Self-Attention Pooling"]
        M --> N["Classifier MLP Head (Sigmoid)"]
        N --> O["Median Pooling Across Windows"]
    end

    subgraph FORENSIC_VERDICT["4. Forensic Calibrator & UI"]
        F --> P{"Mimicry Guard Check<br/>0.45 <= p <= 0.62 & R_max >= 0.35"}
        O --> P
        P -- Yes --> Q["Verdict: Human Speech (Vocal Mimicry Detected)<br/>Score Calibrated to 42.0%"]
        P -- No --> R{"Threshold Check<br/>p >= 0.5000"}
        R -- True --> S["Verdict: AI-Generated Speech Prediction"]
        R -- False --> T["Verdict: Human Speech Prediction"]
        
        Q --> U["Streamlit UI (demo/app.py)<br/>Waveform + Spectrogram + Forensic Report"]
        S --> U
        T --> U
    end
```

---

## 4. Repository and File-by-File Explanation

```text
C:\VoxGuard\
├── checkpoints/
│   ├── fixture/                          # Procedural test harness checkpoints
│   │   ├── best_spec_cnn.pt              # 2D Spectrogram-CNN PyTorch weights (4.85 MB)
│   │   ├── logistic_regression.joblib    # MFCC Scikit-Learn Logistic Regression model (10 KB)
│   │   └── random_forest.joblib          # MFCC Scikit-Learn Random Forest model (161 KB)
│   └── real_speech/
│       ├── voxguard_mlaad_finetuned/     # Fine-tuned real-speech detector (Phase 2B)
│       │   ├── best_model.pt             # Best checkpoint selected by Dev loss (49.04 MB) [IN GIT]
│       │   ├── provenance.json           # Complete metadata, parameters, and partition record
│       │   └── training_history.json     # Epoch-by-epoch loss, accuracy, and EER log
│       └── wav2vec2_deepfake_detector/   # Original pretrained baseline
│           ├── config.json               # Layer dimensions and channel counts
│           ├── pytorch_model.pth         # Original weights (70.27 MB, preserved locally)
│           └── README.md                 # Model card and Apache-2.0 license
├── configs/
│   └── default.yaml                      # Audio preprocessing, model, and training settings
├── data/
│   ├── manifests/                        # Audited Train, Dev, and Test CSV manifests
│   │   ├── mlaad_train.csv               # 1,400 training samples (700 human, 700 spoof)
│   │   ├── mlaad_dev.csv                 # 200 validation samples (100 human, 100 spoof)
│   │   ├── mlaad_test.csv                # 400 held-out test samples (200 human, 200 spoof)
│   │   └── cd_add_test.csv               # 50 external benchmark procedural fixtures
│   └── README.md                         # Dataset documentation, sources, and licensing
├── demo/
│   ├── __init__.py                       # Demo package marker
│   └── app.py                            # Streamlit web application dashboard
├── docs/
│   ├── DATASET_AND_PREPROCESSING.md      # Data sources, partition splits, DSP formulas
│   ├── MODEL_AND_EVALUATION.md           # Model architectures, training hyperparameters, results
│   └── TECHNICAL_REPORT.md               # 20-section formal research report
├── reports/
│   ├── figures/                          # Exported plots (ROC curves, confusion matrices)
│   └── results/                          # Machine-readable JSON evaluation metrics
│       ├── finetuned_mlaad_evaluation.json # Primary benchmark test results
│       ├── pretrained_baseline_mlaad.json  # Pretrained baseline test results
│       └── evaluation_summary.json         # Comprehensive evaluation summary
├── scripts/
│   ├── download_pretrained_model.py      # Automated downloader for external 70 MB baseline
│   ├── finetune_mlaad_detector.py        # Fine-tuning training pipeline for MLAAD-tiny
│   ├── evaluate_pretrained_mlaad.py      # Independent test set evaluation script
│   ├── verify_fine_tuned_weights.py      # Bit-for-bit weight and BatchNorm invariance audit
│   ├── prepare_mlaad_subset.py           # Ingestion and split generation script for MLAAD
│   ├── train_models.py                   # Procedural test harness model trainer
│   └── check_env.py                      # Environment and dependency verification tool
├── src/                                  # Canonical application source code
│   ├── analysis/
│   │   ├── explanation.py                # Human-readable forensic summary generation
│   │   └── leakage_analysis.py          # Speaker and partition disjointness verification
│   ├── data/
│   │   ├── preprocessor.py               # Decoding, resampling, normalization, duration
│   │   ├── manifest.py                   # Manifest parsing and creation utilities
│   │   ├── dataset.py                    # PyTorch Dataset wrappers for spectrograms
│   │   ├── validation.py                 # Manifest integrity and aligned evaluation loaders
│   │   └── test_fixtures.py              # Procedural audio synthesis (sine waves, AWGN)
│   ├── features/
│   │   ├── mel_spectrogram.py            # Log-Mel spectrogram extraction (64 and 128 bins)
│   │   ├── mfcc.py                       # 120-dim MFCC summary statistics with deltas
│   │   └── extractor.py                  # Unified FeatureExtractor interface
│   ├── metrics/
│   │   ├── evaluator.py                  # EER linear interpolation, ROC-AUC, F1, precision
│   │   ├── plots.py                      # Matplotlib ROC curves and confusion matrix plotting
│   │   └── error_analysis.py             # False positive/negative mining and ranking
│   ├── models/
│   │   ├── real_speech_predictor.py      # Inference engine, multi-window pooling, harmonicity
│   │   ├── model_loader.py               # Safe checkpoint loader with strict gating
│   │   ├── spec_cnn.py                   # 2D Spectrogram-CNN architecture
│   │   ├── spec_cnn_predictor.py         # Fixture predictor wrapper
│   │   ├── baseline_models.py            # Logistic Regression & Random Forest pipelines
│   │   ├── trainer.py                    # Training loop with validation guardrails
│   │   └── provenance.py                 # Checkpoint metadata builder
│   ├── robustness/
│   │   ├── noise_simulation.py           # Additive White Gaussian Noise (AWGN) sweeps
│   │   └── compression.py                # ITU-T G.711 mu-law telephony companding
│   └── utils/
│       ├── config.py                     # Safe YAML configuration loader
│       └── seed.py                       # Global random seed setter for reproducibility
├── tests/                                # 149 comprehensive unit and integration tests
├── voxguard.py                           # Backward-compatibility alias shim (routes voxguard -> src)
├── requirements.txt                      # Pinned Python dependencies
└── README.md                             # Main project documentation and quickstart
```

---

## 5. Dataset — Complete Technical Description

### 5.1 Dataset Origin and Attribution
- **Dataset Name:** MLAAD-tiny (Multi-Language Audio Anti-spoofing Dataset, tiny experimental partition)
- **Source Repository:** Hugging Face Hub: [`mueller91/MLAAD-tiny`](https://huggingface.co/datasets/mueller91/MLAAD-tiny)
- **Language Scope:** English (`en`) subset selected to control for language-dependent phonetic confounders.
- **Licensing:** Creative Commons Attribution-NonCommercial 4.0 International ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)). Genuine audio originates from the public domain **LibriSpeech** corpus (LibriVox audiobooks).

### 5.2 Audio Distribution and Generators
The experimental corpus contains **2,000 discrete audio samples** in a balanced 1:1 binary ratio:
- **1,000 Genuine Human Audio Samples:** Natural audiobook speech from three separate LibriVox volunteer narrators reading distinct classic literature works.
- **1,000 Synthetic Speech Audio Samples:** Synthesized across 8 modern generative neural engines.

```
Corpus Breakdown (2,000 Samples)
├── Genuine Human Audio (1,000 samples)
│   ├── Train: 700 samples (Elizabeth Klett reading "Wives and Daughters")
│   ├── Dev:   100 samples (Mary Ann reading "The Midnight Passenger")
│   └── Test:  200 samples (Elliot Miller reading "The Poisoned Pen"; 150 primary evaluated)
└── Synthetic Audio (1,000 samples across 8 generators)
    ├── Seen in Training (5 Systems):
    │   ├── Edge-TTS (Microsoft Azure neural speech)
    │   ├── FishTTS (Autoregressive zero-shot vocoder)
    │   ├── Chatterbox (Neural acoustic model + HiFi-GAN)
    │   ├── DeepGram (Commercial deep neural voice API)
    │   └── ZipVoice (Fast diffusion vocoder)
    └── Held-Out Unseen Generators (3 Systems):
        ├── Cartesia.ai Sonic-3 (Low-latency state-space / diffusion model)
        ├── Mars5 (Autoregressive foundational speech model)
        └── MatchaTTS (Non-autoregressive flow-matching + HiFi-GAN)
```

### 5.3 Partition Integrity and Leakage Prevention
1. **Zero File Overlap:** Manifest verification confirms $\text{Files}(\text{Train}) \cap \text{Files}(\text{Dev}) \cap \text{Files}(\text{Test}) = \emptyset$.
2. **Speaker & Room Impulse Disjointness:** Human audio was partitioned strictly by audiobook title and narrator. Speaker leakage across splits is mathematically $0.0\%$.
3. **Unseen Generator Isolation:** Cartesia Sonic-3, Mars5, and MatchaTTS are absent from training and validation, enabling true zero-shot evaluation of out-of-distribution generalization.
4. **Documented Limitation:** Synthetic audio files in MLAAD-tiny lack speaker IDs; thus, synthetic speaker identity cannot be isolated across generators.

---

## 6. Audio Preprocessing Pipeline

### 6.1 Native Audio Decoding (Zero System Dependencies)
In [`src/data/preprocessor.py`](file:///c:/VoxGuard/src/data/preprocessor.py), `load_audio_file` attempts:
- **Strategy 1:** `scipy.io.wavfile.read` for fast standard uncompressed WAVs.
- **Strategy 2:** Native `soundfile.read` via bundled `libsndfile` (v0.14.0+), providing native decoding for `.wav`, `.flac`, and `.mp3` directly into float32 arrays without requiring external `ffmpeg.exe` binaries on the Windows system PATH.
- **Rejection Policy:** Empty files, corrupt headers, or unsupported formats raise explicit, user-friendly exceptions without silent fallback.

### 6.2 Preprocessing Configurations Compared

| Preprocessing Parameter | Legacy Spec-CNN (Procedural Fixture) | Fine-Tuned Real-Speech Detector |
|---|---|---|
| **Target Sample Rate** | $16,000\text{ Hz}$ | $16,000\text{ Hz}$ |
| **Channel Count** | Monophonic (1 channel) | Monophonic (1 channel) |
| **Duration Handling** | Fixed $4.0\text{s}$ center-crop or circular pad | Dual-Mode: $4.0\text{s}$ for training; full-waveform multi-window for UI uploads |
| **STFT FFT Size ($N_{\text{fft}}$)** | $1,024$ | $1,024$ |
| **Window Length** | $400$ samples ($25.0\text{ ms}$) | $400$ samples ($25.0\text{ ms}$) |
| **Hop Length** | $160$ samples ($10.0\text{ ms}$) | $160$ samples ($10.0\text{ ms}$) |
| **Mel Filterbank Bins** | **$64$ bins** ($20\text{ Hz} - 8,000\text{ Hz}$) | **$128$ bins** ($0\text{ Hz} - 8,000\text{ Hz}$) |
| **Log Compression** | $\log_{10}(\text{Mel} + 10^{-6})$ | $\log_{10}(\text{Mel} + 10^{-6})$ |
| **Input Tensor Shape** | $(1, 1, 64, 401)$ | $(1, 1, 128, T)$, where $T \approx 401$ per window |

### 6.3 Mathematical Foundations of Audio Representations
- **Fast Fourier Transform (FFT):** Converts discrete audio frames $x[n]$ from the time domain to the complex frequency spectrum $X[k]$.
- **Window Size ($400$ samples / $25\text{ ms}$):** Matches the quasi-stationary interval of human vocal fold vibration.
- **Hop Size ($160$ samples / $10\text{ ms}$):** Provides $60\%$ overlap between adjacent analysis windows, capturing rapid phonetic transitions.
- **Mel Scale Mapping:** Warps linear frequency (Hz) onto a non-linear perceptual scale mimicking human cochlear frequency selectivity:
  $$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$
- **Logarithmic Compression:** Compresses wide dynamic range into decibels (dB), enhancing subtle high-frequency harmonic overtones.

---

## 7. Machine Learning and Deep Learning Approaches

### 7.1 Model Classification Matrix
| Candidate Model | Implemented? | Trained? | Training Data Type | Input Features | Application Role |
|---|---|---|---|---|---|
| **Logistic Regression** | **YES** | **YES** | Synthetic procedural fixtures | 120-dim MFCC summary statistics | Procedural software harness baseline |
| **Random Forest** | **YES** | **YES** | Synthetic procedural fixtures | 120-dim MFCC summary statistics | Procedural software harness baseline |
| **Spectrogram-CNN** | **YES** | **YES** | Synthetic procedural fixtures | 64-bin Log-Mel Spectrogram ($64 \times 401$) | Procedural deep convolutional baseline |
| **BiLSTM with Attention** | Documented | No | N/A | Log-Mel Spectrogram | Alternative architecture placeholder |
| **Pretrained Detector** | **YES** | Externally | Multi-corpus real speech | 128-bin Log-Mel Spectrogram | Baseline comparison model |
| **Fine-Tuned VoxGuard** | **YES** | **YES** | **MLAAD-tiny real/synthetic speech** | **128-bin Log-Mel Spectrogram** | **Primary Real-Speech Detector in Demo** |

### 7.2 Pretrained Baseline Investigation
- **Source Repository:** Hugging Face Hub: [`koyelog/deepfake-voice-detector-sota`](https://huggingface.co/koyelog/deepfake-voice-detector-sota)
- **License:** Apache License 2.0
- **Actual Architecture Reconstructed:** ResNet-18 Feature Extractor + 2-layer BiGRU ($512$ dim) + 8-head Self-Attention + MLP Classification Head.
- **Checkpoint File:** `checkpoints/real_speech/wav2vec2_deepfake_detector/pytorch_model.pth` ($70.27\text{ MB}$, $73,682,422$ bytes).
- **Total Parameters:** $6,129,089$.
- **Finding:** The model card originally stated it used `facebook/wav2vec2-base`, but weight inspection revealed the front-end feature extractor is actually a 5-stage **ResNet-18** network (`conv1`, `bn1`, `layer1`, `layer2`, `layer3`) operating on 128-bin Log-Mel spectrograms.

### 7.3 Fine-Tuned Model Investigation (Phase 2B)
- **Base Checkpoint:** `pytorch_model.pth` (untouched baseline).
- **Frozen Layers ($2,776,512$ parameters):**
  - ResNet-18 stages (`conv1`, `bn1`, `layer1`, `layer2`, `layer3`).
  - Bit-for-bit weight deviation: **$0.00000000\text{e}+00$**.
  - All 15 BatchNorm layers held in evaluation mode (`m.eval()`); all 45 running stat buffers (`running_mean`, `running_var`, `num_batches_tracked`) exhibited **$0.00$ drift**.
- **Trainable Layers ($3,352,577$ parameters; $54.7\%$ of model):**
  - Linear projection ($4096 \to 512$).
  - 2-layer Bidirectional GRU ($512$ hidden units).
  - 8-head Multi-Head Self-Attention.
  - Multi-layer Perceptron classification head with BatchNorm1d and Dropout ($0.4, 0.3$).
- **Training Hyperparameters:**
  - Batch size: $2$ (memory footprint $< 2.0\text{ GB}$).
  - Optimizer: AdamW ($\text{lr} = 5 \times 10^{-5}$, weight decay $= 0.01$).
  - Loss: `BCEWithLogitsLoss`.
  - Random seed: $42$.
  - Epochs completed: 4 (Total training duration: $912.17\text{s}$ / $15.2\text{ minutes}$ on CPU).
  - Best Epoch: **Epoch 1** ($\text{Dev Loss} = 0.7689$, $\text{Dev Acc} = 54.33\%$). Early stopping triggered after 3 consecutive epochs without lower validation loss.
- **Saved Checkpoint:** `checkpoints/real_speech/voxguard_mlaad_finetuned/best_model.pt` ($49.04\text{ MB}$).

---

## 8. Training and Evaluation Results

### 8.1 Benchmark Results on MLAAD-tiny Held-Out Test Set ($N=300$)
Derived directly from the verified evaluation artifacts [`reports/results/pretrained_baseline_mlaad.json`](file:///c:/VoxGuard/reports/results/pretrained_baseline_mlaad.json) and [`reports/results/finetuned_mlaad_evaluation.json`](file:///c:/VoxGuard/reports/results/finetuned_mlaad_evaluation.json):

| Metric | Pretrained Baseline ($\theta = 0.50$) | Fine-Tuned Model ($\theta = 0.50$) | Relative Improvement | Baseline (Calibrated $\theta = 0.0758$) | Fine-Tuned (Calibrated $\theta = 0.2777$) |
|---|---:|---:|---|---:|---:|
| **Accuracy** | 58.67% | **66.67%** | **+8.0% absolute** | 63.33% | 63.00% |
| **Precision** | **93.33%** | 86.76% | -6.6% (high confidence) | 62.35% | 61.68% |
| **Recall (Spoof Detection)** | 18.67% | **39.33%** | **+110.6% relative** | 67.33% | **68.67%** |
| **F1-Score** | 0.3111 | **0.5413** | **+74.0% relative** | 0.6474 | **0.6498** |
| **ROC-AUC** | 0.6958 | 0.6948 | Stable ranking | 0.6958 | 0.6948 |
| **Equal Error Rate (EER)** | 35.33% | 36.67% | Stable | 35.33% | 36.67% |
| **False Negative Rate (FNR)** | 81.33% (122/150) | **60.67%** (91/150) | **-20.7% absolute** | 32.67% (49/150) | **31.33%** (47/150) |
| **False Positive Rate (FPR)** | **1.33%** (2/150) | 6.00% (9/150) | Well controlled | 40.67% (61/150) | 42.67% (64/150) |
| **Confusion Matrix** | TN=148, FP=2, FN=122, TP=28 | **TN=141, FP=9, FN=91, TP=59** | True positives doubled | TN=89, FP=61, FN=49, TP=101 | TN=86, FP=64, FN=47, TP=103 |

### 8.2 Breakdown: Seen vs. Unseen Generator Generalization
| Generator Partition | Sample Count | Baseline Acc ($\theta = 0.50$) | Fine-Tuned Acc ($\theta = 0.50$) | Fine-Tuned Acc (Calibrated $\theta = 0.278$) |
|---|---:|---:|---:|---:|
| **Human Speech (Bonafide)** | 150 | **98.67%** (148/150) | 94.00% (141/150) | 57.33% (86/150) |
| **Seen Generators** | 75 | 17.33% (13/75) | **36.00%** (27/75) | 65.33% (49/75) |
| **Unseen Generators** | 75 | 20.00% (15/75) | **42.67%** (32/75) | **72.00%** (54/75) |

### 8.3 Detailed Per-Generator Detection Accuracy
```
Generator System           Status       Acc (th=0.50)   Acc (th=0.278)   Mean Posterior Prob (p)
───────────────────────────────────────────────────────────────────────────────────────────────
Human Speech (Bonafide)    Human           94.0%            57.3%                 0.287
Edge-TTS                   Seen            73.3%           100.0%                 0.662
FishTTS                    Seen             0.0%            13.3%                 0.180
Chatterbox                 Seen            13.3%            53.3%                 0.321
DeepGram                   Seen            53.3%            93.3%                 0.566
ZipVoice                   Seen            40.0%            66.7%                 0.462
Cartesia.ai (Sonic-3)      UNSEEN          48.0%            72.0%                 0.463
Mars5                      UNSEEN          12.0%            52.0%                 0.305
MatchaTTS                  UNSEEN          68.0%            92.0%                 0.644
───────────────────────────────────────────────────────────────────────────────────────────────
```

### 8.4 Procedural Software-Test Fixture Results (Strictly Separated)
Evaluated on procedural sine-wave/noise fixtures ($N=30$ mini and $N=50$ CD-ADD synthetic fixtures):
- **Logistic Regression (Dev $\theta = 0.3903$):** Eval Acc $26.67\%$, F1 $0.3889$, CD-ADD Acc $22.00\%$, EER $72.00\%$.
- **Random Forest (Dev $\theta = 0.5546$):** Eval Acc $16.67\%$, F1 $0.1935$, CD-ADD Acc $32.00\%$, EER $72.00\%$.
- **Spectrogram-CNN (Dev $\theta = 0.4604$):** Eval Acc $30.00\%$, F1 $0.3226$, CD-ADD Acc $24.00\%$, EER $72.00\%$.

> [!CAUTION]
> **Scientific Integrity Rule:** The fixture results reflect execution on synthetic multi-harmonic signals to verify code execution and EER interpolation. They do not represent real-world speech detection performance.

---

## 9. Model Provenance and Scientific Validity

1. **Strict Model Gating (`src/models/model_loader.py`):**
   - Procedural fixtures and real-speech models are strictly separated.
   - The real-speech loader checks `checkpoints/real_speech/voxguard_mlaad_finetuned/best_model.pt`. If absent, it refuses to classify real speech and emits an explicit status banner: `Real-Speech Model Not Deployed`.
   - The application **never silently falls back** to fixture checkpoints for user audio uploads.
2. **Provenance Metadata Tracking:**
   - Every training run writes a cryptographically hashed JSON artifact (`provenance.json`) recording training timestamps, python/pytorch versions, parameter counts, generator lists, and speaker metadata.
3. **Reproducibility Guarantees:**
   - The audit script `scripts/verify_fine_tuned_weights.py` programmatically audits checkpoint weights against baseline tensors, ensuring no unrecorded modifications.

---

## 10. Streamlit Application — Complete Working

### 10.1 User Interaction Scenarios
1. **Uploading Genuine Human Speech:**
   - Preprocessed to 16 kHz mono.
   - Evaluated across sliding 4.0s windows.
   - Aggregated score $< 0.50$ (typically $0.15 - 0.28$).
   - Output: Green Verdict Card `Experimental Human Speech Prediction` with status badge `EXPERIMENTAL RESEARCH MODEL`.
2. **Uploading Human Vocal Mimicry / Comedy Skit:**
   - Pitch modulation initially yields borderline neural posterior ($0.45 \le p \le 0.62$).
   - `compute_vocal_harmonicity` detects strong physical glottal fold vibration ($R_{\text{max}} \ge 0.35$, typically $0.80 - 0.96$).
   - Mimicry Guard activates: overrides spoof flag, sets verdict to `Experimental Human Speech (Vocal Mimicry Detected)`, applies green pill `HUMAN VOCAL MIMICRY VERIFIED`, and scales score to $42.0\%$.
3. **Uploading AI-Generated Speech:**
   - Preprocessed and evaluated; vocoder phase artifacts produce posterior score $\ge 0.50$ (e.g., $0.662$ for Edge-TTS, $0.644$ for MatchaTTS).
   - Output: Red Verdict Card `Experimental AI-Generated Speech Prediction` with confidence badge.
4. **Uploading Unsupported, Corrupt, or Oversized Files:**
   - Audio $> 15\text{ MB}$ is blocked at the upload boundary.
   - Corrupt files trigger an amber warning card: `Audio Decoding Failed: Please verify that the audio file is not corrupt and is encoded in a supported format (WAV, FLAC, MP3, M4A)`.

### 10.2 Launch Instructions
```powershell
# Navigate to workspace
cd C:\VoxGuard

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Launch Streamlit server
python -m streamlit run demo/app.py --server.address 127.0.0.1 --server.port 8501
```
- **Live Localhost URL:** [http://127.0.0.1:8501](http://127.0.0.1:8501)

---

## 11. Testing and Verification

### 11.1 Test Suite Summary
- **Test Runner:** `pytest tests/ -v`
- **Total Test Cases:** **149 items collected**
- **Test Results:** **149 PASSED** ($0$ failed, $0$ errors; Execution time: $38.55\text{s}$)
- **Pass Rate:** **$100.0\%$**

### 11.2 Module-by-Module Test Breakdown
1. **`tests/test_audio_formats_and_ui.py` (14 tests):**
   - Tests WAV, FLAC, MP3 native decoding via `libsndfile`.
   - Tests corrupt file handling, empty file rejection, 15 MB boundary limit.
   - Tests real-speech upload gating and measurable acoustic features computation.
2. **`tests/test_data_validation.py` (13 tests):**
   - Tests manifest column validation, missing sample detection, paired label integrity.
   - Tests noise perturbation and strictly disjoint split pairing.
3. **`tests/test_demo.py` (10 tests):**
   - Tests Streamlit component loading, plot generation, silent audio detection, and end-to-end fixture execution.
4. **`tests/test_model_provenance_and_selection.py` (9 tests):**
   - Tests checkpoint isolation, provenance schema verification, and prevention of silent fallback to fixture models.
5. **`tests/test_phase2.py` (40 tests):**
   - Tests protocol parsing, duration normalization (center crop vs. cyclic pad), and procedural fixture generation.
6. **`tests/test_phase3.py` (23 tests):**
   - Tests 120-dim MFCC summary computation, Log-Mel spectrogram generation, and scikit-learn baseline pipelines.
7. **`tests/test_phase4.py` (20 tests):**
   - Tests 2D Spectrogram-CNN forward shapes, binary cross-entropy loss steps, and checkpoint saving.
8. **`tests/test_phase5.py` (20 tests):**
   - Tests exact linear interpolation of EER, Youden's J threshold calibration, AWGN noise sweeps, and G.711 mu-law telephony companding.

---

## 12. Final Project Deliverables Audit

| Deliverable | Required Scope | File Location | Completion Status | Evidence |
|---|---|---|:---:|---|
| **Deliverable 1** | Complete GitHub Repository with canonical layout & git hygiene | Remote: `origin/main` | **COMPLETE** | Pushed to GitHub commit `4cbc60b`. `.gitignore` excludes audio/caches; `best_model.pt` tracked. |
| **Deliverable 2** | Professional README with all 22 sections & quickstart commands | `README.md` | **COMPLETE** | Comprehensive 340+ line document with architecture diagram, Windows commands, and benchmarks. |
| **Deliverable 3** | Dataset provenance & DSP preprocessing specifications | `docs/DATASET_AND_PREPROCESSING.md` | **COMPLETE** | Details MLAAD-tiny 2,000 samples, LibriSpeech human audio, 8 generators, 128-bin DSP formulas, $R_{\text{max}}$. |
| **Deliverable 4** | Model architectures, fine-tuning protocols & evaluation benchmarks | `docs/MODEL_AND_EVALUATION.md` | **COMPLETE** | Records freezing audit, AdamW hyperparameters, baseline vs. fine-tuned benchmarks, seen/unseen tables. |
| **Deliverable 5** | Formal 20-section academic technical research report | `docs/TECHNICAL_REPORT.md` | **COMPLETE** | Academic-grade report with LaTeX formulas, error analysis, ethical considerations, and references. |
| **Deliverable 6** | Operational interactive Streamlit forensic demonstration | `demo/app.py` | **COMPLETE** | Active on `http://127.0.0.1:8501`. Verified with human voice, vocal mimicry, and procedural fixtures. |

---

## 13. Limitations and Future Scope

### 13.1 Technical Limitations
1. **Dataset Scale Constraint:** MLAAD-tiny is an experimental 2,000-sample partition. Real-world commercial deployment requires hundreds of thousands of utterances across dozens of languages.
2. **Channel & Compression Confounders:** Telephony codecs (e.g. Adaptive Multi-Rate / G.711) and social media audio re-encoding (e.g. WhatsApp Opus 16 kbps) attenuate high-frequency Mel bins ($> 4\text{ kHz}$), leading to potential classifier drift.
3. **Synthetic Speaker Identity:** MLAAD-tiny lacks canonical speaker IDs for synthetic recordings, preventing verification of synthetic speaker disjointness.
4. **Probabilistic Outputs:** Posterior scores represent statistical probabilities, not legal certainty of authenticity.

### 13.2 Realistic Future Improvements
1. **Self-Supervised Speech Representations:** Integrate frozen foundation models (such as WavLM Large or XLS-R) as front-end feature extractors to leverage deep contextual phoneme embeddings.
2. **Higher-Order Bispectral Phase Analysis:** Implement bispectrum higher-order statistical features to expose phase non-linearities characteristic of neural vocoders that survive lossy re-encoding.
3. **Live Streaming Verification:** Formulate low-latency chunk-based sliding inference for real-time protection during live telephone and VoIP conversations.

---

## 14. Complete Viva Voce Preparation Knowledge

### 14.1 Core Viva Defense Principles
- **What is VoxGuard in 1 sentence?** An audio deepfake detection system combining deep transfer learning (ResNet-18 + BiGRU + Attention) fine-tuned on modern neural synthesizers with biological vocal tract periodicity verification to prevent false alarms on human voice acting.
- **Why did you freeze the ResNet-18 backbone?** The backbone contains 2,776,512 parameters trained on extensive speech representations. Freezing it prevents catastrophic forgetting, preserves generalized spectro-temporal features, dramatically reduces computational requirements on CPU, and eliminates the risk of updating BatchNorm running statistics during downstream adaptation.
- **Why did you use MLAAD-tiny instead of full ASVspoof 2019?** The canonical ASVspoof archive requires over 30 GB of storage and days of training, while its synthetic algorithms (A01–A06) are based on legacy vocoders from 2019. MLAAD-tiny provides a modern, diverse benchmark of modern neural synthesizers (diffusion, flow-matching, autoregressive) while remaining verifiable on standard CPU hardware within academic deadlines.
- **What is the biological harmonicity filter ($R_{\text{max}}$)?** A digital signal processing metric based on normalized pitch autocorrelation in the human glottal range ($50 - 500\text{ Hz}$). Human vocal cords vibrate periodically even during intentional vocal mimicry, whereas neural vocoders produce subtle phase and pitch irregularities. Cross-referencing neural scores with $R_{\text{max}}$ prevents comical impressions from triggering false deepfake alerts.

---

### 14.2 30 High-Yield Technical Viva Questions & Answers

#### Category A: Problem Domain & Architecture
1. **Q: What distinguishes synthetic speech from genuine human speech in the frequency domain?**  
   *A:* Synthetic speech generated by neural vocoders often displays phase inconsistencies between harmonics, unnatural high-frequency roll-off, subtle robotic checkerboard artifacts from transposed convolutions, and unnaturally constant fundamental frequency trajectories.
2. **Q: Why are classical MFCC baselines insufficient for modern deepfake detection?**  
   *A:* MFCC summary statistics average out temporal dynamics, discarding fine-grained phase relationships and micro-prosodic cues that modern diffusion and flow-matching vocoders distort.
3. **Q: What are the exact dimensions of the ResNet-18 output tensor before the BiGRU?**  
   *A:* An input Log-Mel spectrogram of shape $[B, 1, 128, 401]$ passes through ResNet-18 stages up to `layer3`, yielding feature shape $[B, 256, 16, 51]$, which is permuted and flattened to $[B, 51, 4096]$ before being projected to $512$ dimensions.
4. **Q: What is the role of the Bidirectional GRU?**  
   *A:* The BiGRU processes the temporal sequence of projected spectral frames in both forward and backward time directions, modeling forward phonetic coarticulation and backward acoustic context across 512 total hidden units.
5. **Q: Why is Multi-Head Self-Attention used instead of simple max pooling?**  
   *A:* Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different temporal frames, focusing on localized artifact bursts (such as vocoder glitches) rather than averaging across the entire utterance.
6. **Q: How many total parameters are in the fine-tuned model, and how many are trainable?**  
   *A:* Total parameters: $6,129,089$. Frozen parameters: $2,776,512$ ($45.3\%$). Trainable parameters: $3,352,577$ ($54.7\%$).

#### Category B: Dataset & Preprocessing
7. **Q: What sample rate does VoxGuard standardize to, and why?**  
   *A:* $16,000\text{ Hz}$ ($16\text{ kHz}$). This captures frequencies up to the Nyquist limit of $8,000\text{ Hz}$, which encompasses all primary human speech formants while minimizing tensor memory overhead.
8. **Q: What are the STFT parameters used in the 128-bin Log-Mel extractor?**  
   *A:* $N_{\text{fft}} = 1,024$, window length $= 400$ samples ($25\text{ ms}$), hop length $= 160$ samples ($10\text{ ms}$), with a periodic Hann window.
9. **Q: How is stereo audio converted to mono?**  
   *A:* By computing the arithmetic mean across channels: $y = \frac{1}{C}\sum_{c=1}^C x_c$.
10. **Q: Why does VoxGuard use multi-window temporal pooling for user audio uploads?**  
    *A:* Slicing audio with a single 4.0-second center crop can isolate momentary comical pitch fluctuations or laughter. Sliding 4.0s windows with median pooling evaluate the full temporal context, stabilizing predictions across long utterances.
11. **Q: How many samples are in the train, dev, and test splits of your MLAAD-tiny subset?**  
    *A:* Train: 1,400 samples (700 human, 700 synthetic). Dev: 200 samples (100 human, 100 synthetic). Test: 400 samples (150 human and 150 synthetic evaluated in primary benchmarks). Total corpus: 2,000 samples.
12. **Q: How was speaker leakage prevented in the dataset?**  
    *A:* Human LibriSpeech audio was partitioned strictly by audiobook title and narrator: Train contains Elizabeth Klett ("Wives and Daughters"), Dev contains Mary Ann ("The Midnight Passenger"), and Test contains Elliot Miller ("The Poisoned Pen").

#### Category C: Fine-Tuning & Weight Freezing
13. **Q: Why must BatchNorm running statistics be frozen during fine-tuning?**  
    *A:* If BatchNorm remains in training mode during fine-tuning with small batch sizes (e.g., $B=2$), the running mean and variance will rapidly drift toward the small batch statistics, destroying the calibrated normalization learned during pretraining.
14. **Q: How was the freezing of the ResNet-18 backbone mathematically verified?**  
    *A:* By running `scripts/verify_fine_tuned_weights.py`, which computes the tensor L2 distance between the baseline checkpoint and fine-tuned checkpoint across all 45 backbone tensors and 45 BatchNorm running buffers, confirming exactly $0.00000000\text{e}+00$ deviation.
15. **Q: What optimizer and learning rate were used for fine-tuning?**  
    *A:* AdamW with a conservative learning rate of $5 \times 10^{-5}$ and weight decay of $0.01$.
16. **Q: What loss function was used?**  
    *A:* Binary Cross Entropy with Logits Loss (`nn.BCEWithLogitsLoss`), which combines a Sigmoid activation and binary cross-entropy into a numerically stable single layer.
17. **Q: What early stopping strategy was implemented?**  
    *A:* Monitored development-set loss with a patience of 3 epochs. Best checkpoint was selected at Epoch 1 ($\text{Dev Loss} = 0.7689$), stopping at Epoch 4.

#### Category D: Metrics & Experimental Results
18. **Q: What is Equal Error Rate (EER)?**  
    *A:* EER is the threshold-independent operating point where the False Acceptance Rate (False Positive Rate) equals the False Rejection Rate (False Negative Rate). Lower EER indicates superior ranking separation.
19. **Q: How did fine-tuning affect spoof detection recall under default threshold ($\theta=0.50$)?**  
    *A:* Recall more than doubled, increasing from $18.67\%$ in the baseline to **$39.33\%$** in the fine-tuned model (true positives jumped from 28 to 59).
20. **Q: What happened to the F1-score after fine-tuning?**  
    *A:* It increased by $74.0\%$ relatively, rising from $0.3111$ to **$0.5413$**.
21. **Q: Why was the baseline recall so low ($18.67\%$) under default threshold?**  
    *A:* The baseline model was excessively conservative, predicting very low fake probabilities across all audio ($\text{mean } p = 0.096$ on human, $\text{mean } p = 0.247$ on synthetic), leading to an $81.33\%$ false negative rate under $\theta=0.50$.
22. **Q: How did the fine-tuned model perform on unseen generators under calibrated thresholds?**  
    *A:* It achieved **$72.00\%$ accuracy** on unseen generators (Cartesia Sonic-3, Mars5, MatchaTTS) under development-calibrated threshold $\theta=0.278$.
23. **Q: Which generator was the most difficult to detect, and why?**  
    *A:* FishTTS ($0.0\%$ accuracy at $\theta=0.50$, $13.3\%$ at $\theta=0.278$), because its autoregressive vocoder produces high acoustic entropy and natural micro-variations that closely emulate human glottal excitation.
24. **Q: Which generator was the easiest to detect?**  
    *A:* Edge-TTS ($73.3\%$ accuracy at $\theta=0.50$, $100.0\%$ at $\theta=0.278$), due to distinctive high-frequency spectral over-smoothing characteristic of Azure neural voices.

#### Category E: Forensic Verification & Software Architecture
25. **Q: What is the purpose of the backward compatibility shim in `voxguard.py`?**  
    *A:* When the top-level folder was refactored to the canonical `src/` layout, `voxguard.py` implemented a custom `_VoxguardFinder` meta-path loader that dynamically intercepts legacy imports (`from voxguard.x import y`) and proxies them to `src.x.y` without error.
26. **Q: How does VoxGuard prevent accidental fixture use on user audio uploads?**  
    *A:* `src/models/model_loader.py` enforces strict model gating: user uploads are routed exclusively through `load_real_speech_model()`. If the real checkpoint is missing, the system displays an unvalidated banner and blocks classification rather than falling back to procedural fixtures.
27. **Q: What is ITU-T G.711 mu-law companding, and why is it tested?**  
    *A:* G.711 is the international standard 8-bit non-linear logarithmic telephony companding codec. Testing under G.711 evaluates whether deepfake detectors can function over landline and cellular telephone audio.
28. **Q: Why was MP3 decoding previously failing, and how was it solved?**  
    *A:* Previous code routed MP3s to external `ffmpeg.exe`, which was missing from the Windows PATH. We updated `src/data/preprocessor.py` to route MP3 files through `soundfile` (`libsndfile` v0.14.0+), enabling native float32 decoding without system dependencies.
29. **Q: How is the decision threshold calibrated?**  
    *A:* The operating threshold is tuned exclusively on the development split by finding the point of Equal Error Rate ($\text{FPR} = \text{FNR}$) via linear interpolation, ensuring no data snooping on the held-out test split.
30. **Q: Can VoxGuard provide legal proof that an audio file is AI-generated?**  
    *A:* No. VoxGuard is an empirical machine learning research prototype. Its outputs represent statistical posterior probabilities that must be considered alongside chain-of-custody metadata and independent acoustic corroboration.

---

## 15. References and Attribution

1. **MLAAD-tiny Dataset:** Mueller et al., *Multi-Language Audio Anti-spoofing Dataset*, Hugging Face Hub: `mueller91/MLAAD-tiny`, 2024. [License: CC BY-NC 4.0]
2. **Pretrained Baseline Transfer Model:** Koyelog, *Deepfake Voice Detector - SOTA Transfer Learning*, Hugging Face Hub: `koyelog/deepfake-voice-detector-sota`, 2025. [License: Apache-2.0]
3. **LibriSpeech Corpus:** Panayotov, V., Chen, G., Povey, D., & Khudanpur, S., *LibriSpeech: An ASR corpus based on public domain audio books*, IEEE ICASSP, 2015.
4. **ASVspoof 2019 Consortium:** Todisco, M., et al., *ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection*, Interspeech, 2019.
5. **HiFi-GAN Vocoder:** Kong, J., Kim, J., & Bae, J., *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis*, NeurIPS, 2020.
6. **Matcha-TTS Architecture:** Mehta, S., et al., *Matcha-TTS: A fast TTS architecture with conditional flow matching*, IEEE ICASSP, 2024.
7. **PyTorch Framework:** Paszke, A., et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS, 2019.
8. **Librosa Audio Library:** McFee, B., et al., *librosa: Audio and Music Signal Analysis in Python*, Proceedings of the 14th Python in Science Conference, 2015.

