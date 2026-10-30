# VoxGuard Data Management & Provenance

This directory outlines the dataset modes, partitioning protocols, storage layout, and governance policies utilized in the VoxGuard deepfake audio detection project.

---

## 1. Operational Modes & Scientific Boundaries

VoxGuard strictly delineates between two operational regimes to prevent test fixtures from masquerading as real-world speech models:

### Mode A: Synthetic Fixture Mode (`dataset_mode: "synthetic_fixture"`)
* **Audio Source:** Procedurally synthesized multi-harmonic sine tones and additive white noise generated deterministically by `tests/fixtures/conftest.py` (`generate_sine_tone()`).
* **Purpose:** Software unit tests, architectural verification, tensor shape validation, feature extraction pipeline debugging, and deterministic continuous integration.
* **Scientific Scope:** Establishes software pipeline correctness, leak-free partition mechanics, and mathematical validity of EER/ROC-AUC routines.
* **Explicit Boundary:** **Must NEVER be represented as measuring real human voice deepfake detection capability.**

### Mode B: Real Speech Research Mode (`dataset_mode: "real_speech"`)
* **Audio Source:** Genuine human speech recordings and authentic generative text-to-speech (TTS) / voice conversion (VC) attacks from the official ASVspoof 2019 Logical Access (LA) database and CD-ADD benchmark.
* **Purpose:** Empirical speech anti-spoofing research, acoustic feature evaluation, and out-of-domain domain shift benchmarking.
* **Enforcement:** The pipeline strictly fails with explicit errors if real speech protocols or audio files are absent, completely blocking silent fixture fallbacks.

---

## 2. Primary Dataset: ASVspoof 2019 Logical Access (LA)

### Overview & Provenance
The primary dataset used for model development is the **ASVspoof 2019 Logical Access (LA)** database. It is the benchmark standard for evaluating speech anti-spoofing countermeasures against state-of-the-art text-to-speech (TTS) synthesis and voice conversion (VC) algorithms.

- **Official Source:** [Edinburgh DataShare - ASVspoof 2019](https://datashare.ed.ac.uk/handle/10283/3336)
- **Consortium Portal:** [ASVspoof Official Website](https://www.asvspoof.org/index2019.html)
- **Primary Reference:**
  > Todisco, M., Wang, X., Vestman, V., Sahidullah, M., Delgado, H., Nautsch, A., Yamagishi, J., Evans, N., Kinnunen, T., & Lee, K. A. (2019). *ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection*. Proc. Interspeech 2019, 1008–1012.

### Partitions & Speaker-Disjoint Structure
The dataset provides official, strictly speaker-disjoint partitions:

| Partition | Total Utterances | Bonafide (Human) | Spoof (Synthetic) | Speakers | Spoofing Algorithms |
|---|---:|---:|---:|---:|---|
| **Train** | 25,380 | 2,580 | 22,800 | 20 (8 M, 12 F) | Known: A01 – A06 |
| **Development (Dev)** | 24,844 | 2,548 | 22,296 | 20 (8 M, 12 F) | Known: A01 – A06 |
| **Evaluation (Eval)** | 71,237 | 7,355 | 63,882 | 67 (31 M, 36 F) | Unseen: A07 – A19 |

*Critical Protocol Note:*
- $\text{Speakers}(\text{Train}) \cap \text{Speakers}(\text{Dev}) = \emptyset$
- $\text{Speakers}(\text{Train}) \cap \text{Speakers}(\text{Eval}) = \emptyset$
- Evaluation algorithms A07–A19 represent unseen voice generation systems (including neural vocoders, waveform concatenation, and advanced VC), serving as an internal unseen-generator generalization test.

### Audio Specifications
- **Format:** FLAC (Free Lossless Audio Codec)
- **Sampling Rate:** 16,000 Hz (16 kHz)
- **Bit Depth:** 16-bit linear PCM
- **Channels:** Single channel (monophonic)

---

## 2. External Evaluation Dataset: CD-ADD

### Overview & Provenance
To rigorously evaluate out-of-distribution generalization, VoxGuard uses the **Cross-Domain Audio Deepfake Detection (CD-ADD)** benchmark.

- **Repository:** [SpeechAntiSpoofingBenchmarks/CD-ADD on Hugging Face](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/CD-ADD)
- **Reference Paper:**
  > arXiv:2404.04904: *Cross-Domain Audio Deepfake Detection: Dataset and Analysis* (2024).

### Strict Evaluation Policy
> [!IMPORTANT]
> **Zero-Contamination Rule:** CD-ADD is strictly an external, held-out evaluation dataset.
> 1. It is **never** used during training.
> 2. It is **never** used to fit preprocessing parameters (e.g., normalizer means or scalers).
> 3. It is **never** used for model selection or decision threshold tuning (EER thresholds are derived exclusively from the ASVspoof dev partition).
> 4. It is evaluated solely to report empirical out-of-domain performance drops.

---

## 3. Directory Layout

The physical data directory is organized into separate stages to prevent contamination:

```
data/
├── README.md               # This provenance document
├── manifests/              # Partition protocol files and metadata manifests
│   ├── asvspoof_train.csv  # Parsed train protocol (speaker, file, attack, label)
│   ├── asvspoof_dev.csv    # Parsed dev protocol
│   ├── asvspoof_eval.csv   # Parsed eval protocol
│   └── cd_add_test.csv     # Parsed external test manifest
├── raw/                    # [GITIGNORED] Raw audio files (.flac, .wav)
│   ├── asvspoof2019_la/
│   │   ├── train/
│   │   ├── dev/
│   │   └── eval/
│   └── cd_add/
│       └── test/
└── processed/              # [GITIGNORED] Extracted feature matrices (.npy, .pt)
    ├── mel_train.pt
    ├── mfcc_train.npy
    ├── mel_dev.pt
    └── mfcc_dev.npy
```

---

## 4. Download & Verification Procedures

### Local Development Subset
For local CPU prototyping and smoke tests without downloading the full ~15 GB archive:
```bash
python scripts/download_data.py --subset mini --samples-per-class 500
```

### Full Dataset Acquisition (Google Colab / Dedicated Machine)
To download the complete ASVspoof 2019 LA partition directly from Edinburgh DataShare:
```bash
python scripts/download_data.py --subset full --output-dir data/raw/asvspoof2019_la
```

To prepare the external CD-ADD test partition:
```bash
python scripts/download_data.py --subset cd_add --split test --output-dir data/raw/cd_add
```

---

## 5. Research Data-Use & Licensing Notice

1. **ASVspoof 2019 LA License:** The ASVspoof 2019 database is made available for non-commercial academic research under the terms of the Open Data Commons Attribution License (ODC-BY). Users must review and adhere to the official consortium terms. Raw audio files must not be redistributed or committed to public Git repositories.
2. **CD-ADD Benchmark Terms:** Users accessing CD-ADD must comply with the license and terms stipulated by the authors on Hugging Face.
3. **Forensic Disclaimer:** Datasets are collected under controlled acoustic conditions. High benchmark scores do not guarantee detection capability against novel in-the-wild deepfakes or social media audio degraded by unknown re-encoding cascades.

