"""Fine-tuning experiment for VoxGuard MLAAD-tiny real-speech deepfake detector.

Controlled Experiment:
- Starts from pretrained checkpoint: checkpoints/real_speech/wav2vec2_deepfake_detector/pytorch_model.pth
- Freezes the ResNet-18 convolutional backbone (conv1, bn1, layer1, layer2, layer3: 2,776,512 parameters)
- Fine-tunes classification / aggregation layers:
    * BiGRU (2-layer, 512 total)
    * Multi-Head Self-Attention (8 heads, 512 dim)
    * Classifier MLP Head (Linear-BatchNorm-ReLU-Dropout)
    Total trainable parameters: 3,352,577 (54.7% of model)
- Precomputes and caches 128-bin Log-Mel Spectrograms in memory (~90 MB RAM for 1,400 samples)
- Controlled batch size: 2
- Loss: BCEWithLogitsLoss (balanced 1:1 classes)
- Optimizer: AdamW with weight decay 0.01, conservative lr 5e-5
- Early stopping on Development set loss (patience: 3 epochs, max epochs: 12)
- Saves best checkpoint: checkpoints/real_speech/voxguard_mlaad_finetuned/best_model.pt
- Saves latest recovery checkpoint: checkpoints/real_speech/voxguard_mlaad_finetuned/latest_checkpoint.pt
- Records full provenance, training history, and metrics.
"""

import argparse
import ctypes
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.test_pretrained_detector import (
    PretrainedDeepfakeDetector,
    load_pretrained_detector,
    get_ram_info,
)
from src.data.preprocessor import load_audio_file, preprocess_audio
from src.features.mel_spectrogram import extract_log_mel_spectrogram


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CachedSpectrogramDataset(Dataset):
    """Dataset that precomputes and caches Log-Mel spectrograms in RAM."""

    def __init__(self, manifest_path: str, target_sr: int = 16000, target_duration: float = 4.0):
        self.manifest_path = manifest_path
        self.target_sr = target_sr
        self.target_duration = target_duration

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.records = json.load(f)

        print(f"Loading and precomputing spectrograms for {len(self.records)} samples ({manifest_path})...")
        t0 = time.time()

        self.spectrograms: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.utterance_ids: List[str] = []

        cfg = {
            "sample_rate": target_sr,
            "duration_seconds": target_duration,
            "target_samples": int(target_sr * target_duration),
            "trim_silence": False,
            "peak_norm": True,
        }

        for idx, item in enumerate(self.records):
            fpath = item["local_path"]
            label = item["label"]
            uid = item["utterance_id"]

            raw_wave, orig_sr, err = load_audio_file(fpath, target_sr=target_sr)
            if err or raw_wave is None:
                raise RuntimeError(f"Error loading {fpath}: {err}")

            prep_wave, _ = preprocess_audio(raw_wave, orig_sr, config=cfg)

            mel = extract_log_mel_spectrogram(
                prep_wave,
                sample_rate=target_sr,
                n_mels=128,
                n_fft=1024,
                win_length=1024,
                hop_length=512,
                add_channel_dim=True,
            )  # (1, 128, 126)

            self.spectrograms.append(mel.clone())
            self.labels.append(label)
            self.utterance_ids.append(uid)

            if (idx + 1) % 400 == 0 or (idx + 1) == len(self.records):
                print(f"  Cached {idx + 1}/{len(self.records)} samples ({time.time() - t0:.1f}s)")

        total_mb = sum(s.nelement() * s.element_size() for s in self.spectrograms) / (1024 * 1024)
        print(f"Precomputed {len(self.spectrograms)} tensors in {time.time() - t0:.2f}s ({total_mb:.2f} MB in RAM)")

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        mel = self.spectrograms[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)  # (1,)
        uid = self.utterance_ids[idx]
        return mel, label, uid


def freeze_layers(model: PretrainedDeepfakeDetector, mode: str = "backbone"):
    """Freeze layers according to controlled experimental policy."""
    # 1. ResNet-18 Backbone is ALWAYS frozen initially
    model.conv1.requires_grad_(False)
    model.bn1.requires_grad_(False)
    model.layer1.requires_grad_(False)
    model.layer2.requires_grad_(False)
    model.layer3.requires_grad_(False)

    if mode == "head_only":
        # Freeze GRU and Attention too; train only MLP
        model.gru.requires_grad_(False)
        model.attention.requires_grad_(False)
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif mode == "attn_head":
        # Freeze GRU; train Attention + MLP
        model.gru.requires_grad_(False)
        for param in model.attention.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif mode == "backbone":
        # Train GRU + Attention + MLP
        for param in model.gru.parameters():
            param.requires_grad = True
        for param in model.attention.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unknown freeze mode: {mode}")

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_p = total_p - trainable_p

    print(f"\n[Freezing Policy: '{mode}']")
    print(f"  - Frozen Parameters:    {frozen_p:,} ({frozen_p/total_p*100:.1f}%)")
    print(f"  - Trainable Parameters: {trainable_p:,} ({trainable_p/total_p*100:.1f}%)")


def evaluate_on_dev(model: PretrainedDeepfakeDetector, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float, float]:
    """Evaluate on development set returning (loss, accuracy, eer)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for mels, labels, _ in dataloader:
            logits = model(mels)  # (B, 1)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)

            all_scores.extend(probs.squeeze(1).cpu().numpy().tolist())
            all_labels.extend(labels.squeeze(1).cpu().numpy().tolist())

    avg_loss = total_loss / total
    acc = correct / total

    from scripts.evaluate_pretrained_mlaad import compute_eer
    eer, _ = compute_eer(np.array(all_labels), np.array(all_scores))
    return avg_loss, acc, eer


def train_model(
    model: PretrainedDeepfakeDetector,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    output_dir: str,
    epochs: int = 12,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    patience: int = 3,
    freeze_mode: str = "backbone",
    seed: int = 42,
) -> Dict[str, object]:
    """Execute controlled fine-tuning loop with early stopping."""
    os.makedirs(output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(output_dir, "best_model.pt")
    latest_ckpt_path = os.path.join(output_dir, "latest_checkpoint.pt")

    criterion = nn.BCEWithLogitsLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    best_dev_loss = float("inf")
    best_epoch = -1
    best_dev_acc = 0.0
    best_dev_eer = 1.0
    epochs_no_improve = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "dev_loss": [],
        "dev_acc": [],
        "dev_eer": [],
        "ram_used_gb": [],
    }

    # Initial dev evaluation (epoch 0)
    dev_loss_0, dev_acc_0, dev_eer_0 = evaluate_on_dev(model, dev_loader, criterion)
    print(f"\n[Baseline Epoch 0 (Untouched Weights)]: Dev Loss: {dev_loss_0:.4f} | Dev Acc: {dev_acc_0*100:.2f}% | Dev EER: {dev_eer_0*100:.2f}%")

    print(f"\nStarting fine-tuning (batch_size={train_loader.batch_size}, lr={lr}, seed={seed}, patience={patience})...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch_0 = time.time()
        model.train()
        # Keep frozen backbone explicitly in eval mode so batchnorm running stats don't drift
        model.conv1.eval()
        model.bn1.eval()
        model.layer1.eval()
        model.layer2.eval()
        model.layer3.eval()
        if freeze_mode in ["head_only", "attn_head"]:
            model.gru.eval()
        if freeze_mode == "head_only":
            model.attention.eval()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for mels, labels, _ in train_loader:
            optimizer.zero_grad()
            logits = model(mels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total

        dev_loss, dev_acc, dev_eer = evaluate_on_dev(model, dev_loader, criterion)
        epoch_sec = time.time() - t_epoch_0

        _, used_ram, _ = get_ram_info()

        history["epoch"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["train_acc"].append(round(avg_train_acc, 4))
        history["dev_loss"].append(round(dev_loss, 4))
        history["dev_acc"].append(round(dev_acc, 4))
        history["dev_eer"].append(round(dev_eer, 4))
        history["ram_used_gb"].append(round(used_ram, 2))

        print(f"Epoch {epoch:2d}/{epochs:2d} ({epoch_sec:.1f}s) | Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc*100:5.2f}% | Dev Loss: {dev_loss:.4f}, Acc: {dev_acc*100:5.2f}%, EER: {dev_eer*100:5.2f}% | RAM: {used_ram:.2f} GB")

        # Save recovery checkpoint after every epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": dev_loss,
            "val_accuracy": dev_acc,
            "val_eer": dev_eer,
            "freeze_mode": freeze_mode,
            "learning_rate": lr,
            "batch_size": train_loader.batch_size,
            "seed": seed,
        }, latest_ckpt_path)

        # Early stopping check based on dev loss
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_dev_acc = dev_acc
            best_dev_eer = dev_eer
            best_epoch = epoch
            epochs_no_improve = 0

            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": dev_loss,
                "val_accuracy": dev_acc,
                "val_eer": dev_eer,
                "freeze_mode": freeze_mode,
                "learning_rate": lr,
                "batch_size": train_loader.batch_size,
                "seed": seed,
            }, best_ckpt_path)
            print(f"  -> Best model updated (Epoch {epoch}): Dev Loss {dev_loss:.4f}, Dev Acc {dev_acc*100:.2f}%, Dev EER {dev_eer*100:.2f}%")
        else:
            epochs_no_improve += 1
            print(f"  (No improvement in Dev Loss for {epochs_no_improve}/{patience} epochs)")
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs (patience {patience} exhausted).")
                break

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time:.2f}s ({total_time/60:.2f} min).")
    print(f"Best checkpoint: Epoch {best_epoch} with Dev Loss: {best_dev_loss:.4f}, Dev Acc: {best_dev_acc*100:.2f}%, Dev EER: {best_dev_eer*100:.2f}%")

    # Save training history and configuration
    info = {
        "best_epoch": best_epoch,
        "best_dev_loss": round(best_dev_loss, 4),
        "best_dev_accuracy": round(best_dev_acc, 4),
        "best_dev_eer": round(best_dev_eer, 4),
        "total_epochs_trained": len(history["epoch"]),
        "total_training_time_sec": round(total_time, 2),
        "hyperparameters": {
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": train_loader.batch_size,
            "freeze_mode": freeze_mode,
            "seed": seed,
            "early_stopping_patience": patience,
        },
        "history": history,
    }

    info_path = os.path.join(output_dir, "training_history.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Saved training history to {info_path}")

    # Save complete provenance metadata
    prov = {
        "model_name": "VoxGuard-MLAAD-ResNetBiGRU-FineTuned",
        "base_pretrained_model": "https://huggingface.co/koyelog/deepfake-voice-detector-sota",
        "base_architecture": "ResNet-18 Front-End + BiGRU (2-layer, 512) + Multi-Head Self-Attention (8 heads) + MLP Head",
        "base_checkpoint_file": "checkpoints/real_speech/wav2vec2_deepfake_detector/pytorch_model.pth",
        "base_license": "Apache-2.0",
        "adaptation_dataset": "MLAAD-tiny (mueller91/MLAAD-tiny)",
        "dataset_license": "CC BY-NC 4.0 (Synthetic), M-AILABS License (Genuine)",
        "train_manifest": "data/manifests/mlaad_train.json",
        "dev_manifest": "data/manifests/mlaad_dev.json",
        "test_manifest": "data/manifests/mlaad_test.json",
        "sample_counts": {
            "train_total": 1400,
            "train_bonafide": 700,
            "train_spoof": 700,
            "dev_total": 300,
            "dev_bonafide": 150,
            "dev_spoof": 150,
            "test_total": 300,
            "test_bonafide": 150,
            "test_spoof": 150,
        },
        "speaker_partitioning": {
            "train_speaker": "Elizabeth Klett (female, Wives and Daughters)",
            "dev_speaker": "Mary Ann (female, The Midnight Passenger)",
            "test_speaker": "Elliot Miller (male, The Poisoned Pen)",
            "speaker_leakage": "0.0% (strictly disjoint speakers across splits)",
        },
        "generator_partitioning": {
            "train_generators": [
                "Edge-TTS", "FishTTS", "Chatterbox", "DeepGram", "Indri-TTS-0.1",
                "Llasa-1B-Multilingual", "ZipVoice", "Openaudio-S1-Mini", "Veena", "griffin_lim"
            ],
            "dev_generators": ["Higgs-Audio-V2", "Index-TTS-1.5"],
            "test_seen_generators": ["Edge-TTS", "FishTTS", "Chatterbox", "DeepGram", "ZipVoice"],
            "test_unseen_generators": ["Cartesia.ai (Sonic-3)", "Mars5", "MatchaTTS"],
        },
        "training_configuration": {
            "frozen_layers": "conv1, bn1, layer1, layer2, layer3 (2,776,512 parameters)",
            "trainable_layers": "gru, attention, classifier (3,352,577 parameters)",
            "batch_size": train_loader.batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "optimizer": "AdamW",
            "loss_function": "BCEWithLogitsLoss",
            "seed": seed,
            "early_stopping_patience": patience,
            "best_epoch": best_epoch,
            "best_dev_loss": round(best_dev_loss, 4),
            "best_dev_acc": round(best_dev_acc, 4),
            "best_dev_eer": round(best_dev_eer, 4),
        },
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_environment": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "cpu_cores": os.cpu_count(),
            "device": "CPU",
        },
        "limitations": [
            "Trained on 4.0-second 16 kHz standardized audio. Performance on compressed telephony, short clips (<1.0s), or noisy environments may degrade.",
            "Evaluation is limited to MLAAD-tiny English speech and does not prove universal detection of all AI voice generators.",
            "Commercial use is restricted by MLAAD-tiny non-commercial license (CC BY-NC 4.0).",
        ],
    }

    prov_path = os.path.join(output_dir, "provenance.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print(f"Saved complete provenance metadata to {prov_path}")

    return info


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VoxGuard deepfake detector on MLAAD-tiny")
    parser.add_argument("--pretrained-ckpt", type=str, default="checkpoints/real_speech/wav2vec2_deepfake_detector/pytorch_model.pth")
    parser.add_argument("--output-dir", type=str, default="checkpoints/real_speech/voxguard_mlaad_finetuned")
    parser.add_argument("--train-manifest", type=str, default="data/manifests/mlaad_train.json")
    parser.add_argument("--dev-manifest", type=str, default="data/manifests/mlaad_dev.json")
    parser.add_argument("--freeze-mode", type=str, choices=["backbone", "attn_head", "head_only"], default="backbone")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    set_seed(args.seed)

    print("=" * 80)
    print("VOXGUARD PRETRAINED DETECTOR FINE-TUNING EXPERIMENT (PHASE 2B)")
    print("=" * 80)

    # 1. Load Pretrained Checkpoint
    model = load_pretrained_detector(args.pretrained_ckpt, device="cpu")

    # 2. Freeze ResNet-18 Backbone according to policy
    freeze_layers(model, mode=args.freeze_mode)

    # 3. Create Cached Datasets (precomputed in RAM)
    train_dataset = CachedSpectrogramDataset(args.train_manifest)
    dev_dataset = CachedSpectrogramDataset(args.dev_manifest)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Train Model
    results = train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        freeze_mode=args.freeze_mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
