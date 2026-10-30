"""Comprehensive tests for VoxGuard Phase 4: 2D Spectrogram-CNN and deepfake audio classification.

Tests cover:
  1. 2D SpectrogramCNN architecture and tensor forward pass shape.
  2. Classification head and calibrated binary probability output.
  3. Strict finite value checks on logits and softmax outputs.
  4. Class imbalance weighting calculation and loss computation.
  5. Training and evaluation loop execution with deterministic seeding.
  6. Model selection and checkpoint persistence (including preprocessing parameters).
  7. Evaluation partition security guardrail (rejecting 'eval' split in training loop).
  8. SpecCNNPredictor single prediction interface and inference consistency.
"""

from pathlib import Path
import sys
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import SpectrogramDataset
from src.data.test_fixtures import generate_sine_tone
from src.models.spec_cnn import ConvBlock, SpectrogramCNN
from src.models.spec_cnn_predictor import SpecCNNPredictor
from src.models.trainer import evaluate_epoch, train_model, train_one_epoch
from src.utils.seed import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# 1. SpectrogramCNN Architecture & Forward Pass Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSpectrogramCNNArchitecture:
    """Tests for 2D SpectrogramCNN layer shapes and forward pass."""

    def test_conv_block_output_shape(self):
        """ConvBlock should apply Conv2d -> BN -> ReLU -> MaxPool2d reducing spatial size by 2."""
        block = ConvBlock(in_channels=1, out_channels=32, kernel_size=3, padding=1, pool_size=2)
        x = torch.randn(2, 1, 64, 400)
        out = block(x)

        assert out.shape == (2, 32, 32, 200), f"Expected (2, 32, 32, 200), got {out.shape}"
        assert torch.isfinite(out).all().item()

    def test_cnn_forward_shape_standard_log_mel(self):
        """Input of shape (B, 1, 64, 401) should output binary logits of shape (B, 2)."""
        model = SpectrogramCNN(
            in_channels=1,
            conv_channels=[32, 64, 128, 256],
            dense_dim=128,
            dropout=0.3,
            num_classes=2,
        )
        x = torch.randn(4, 1, 64, 401)
        logits = model(x)

        assert logits.shape == (4, 2), f"Expected logits shape (4, 2), got {logits.shape}"
        assert torch.isfinite(logits).all().item()

    def test_cnn_forward_with_3d_input(self):
        """Input without channel dim (B, 64, 401) should auto-expand to (B, 1, 64, 401)."""
        model = SpectrogramCNN(in_channels=1)
        x = torch.randn(2, 64, 401)
        logits = model(x)

        assert logits.shape == (2, 2)
        assert torch.isfinite(logits).all().item()

    def test_predict_proba_calibrated_probabilities(self):
        """predict_proba must return valid probabilities in [0.0, 1.0] summing to 1.0."""
        model = SpectrogramCNN()
        x = torch.randn(3, 1, 64, 401)
        probs = model.predict_proba(x)

        assert probs.shape == (3, 2)
        assert (probs >= 0.0).all().item()
        assert (probs <= 1.0).all().item()
        sums = probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(3), atol=1e-5, rtol=1e-5)

    def test_get_config_dictionary(self):
        """get_config must return all architecture hyperparameters."""
        model = SpectrogramCNN(conv_channels=[16, 32], dense_dim=64, dropout=0.2)
        cfg = model.get_config()

        assert cfg["model_type"] == "SpectrogramCNN"
        assert cfg["in_channels"] == 1
        assert cfg["conv_channels"] == [16, 32]
        assert cfg["dense_dim"] == 64
        assert cfg["dropout"] == 0.2
        assert cfg["num_classes"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# 2. SpectrogramDataset & Class Imbalance Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSpectrogramDatasetAndImbalance:
    """Tests for dataset tensor indexing and inverse-frequency class weighting."""

    def test_dataset_len_and_getitem(self):
        """SpectrogramDataset should index samples and return (tensor, label)."""
        specs = torch.randn(10, 1, 64, 401)
        labels = torch.tensor([0, 1] * 5)
        ds = SpectrogramDataset(specs, labels)

        assert len(ds) == 10
        x, y = ds[0]
        assert x.shape == (1, 64, 401)
        assert y.item() == 0

    def test_dataset_adds_channel_dim_if_3d(self):
        """3D spectrogram input (N, 64, 401) should have channel dim added: (N, 1, 64, 401)."""
        specs = torch.randn(5, 64, 401)
        labels = torch.zeros(5, dtype=torch.int64)
        ds = SpectrogramDataset(specs, labels)

        x, _ = ds[0]
        assert x.shape == (1, 64, 401)

    def test_compute_class_weights_imbalanced(self):
        """Inverse-frequency weights must penalize the minority class proportionally."""
        # 9:1 imbalance (10 bonafide, 90 spoof = 100 total)
        labels = torch.tensor([0] * 10 + [1] * 90, dtype=torch.int64)
        specs = torch.randn(100, 1, 64, 401)
        ds = SpectrogramDataset(specs, labels)

        weights = ds.compute_class_weights()
        assert weights.shape == (2,)
        # Expected: w_0 = 100 / (2 * 10) = 5.0, w_1 = 100 / (2 * 90) = 0.555...
        assert abs(weights[0].item() - 5.0) < 1e-4
        assert abs(weights[1].item() - (100.0 / 180.0)) < 1e-4
        assert weights[0].item() > weights[1].item()  # Minority class has higher weight


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training Loop, Validation & Model Selection Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingEngine:
    """Tests for training loop execution, validation, and checkpoint serialization."""

    @pytest.fixture
    def small_datasets(self):
        """Create small synthetic datasets for fast smoke testing."""
        set_seed(42)
        X_train = torch.randn(16, 1, 64, 100)
        y_train = torch.tensor([0] * 4 + [1] * 12, dtype=torch.int64)

        X_dev = torch.randn(8, 1, 64, 100)
        y_dev = torch.tensor([0] * 3 + [1] * 5, dtype=torch.int64)

        return SpectrogramDataset(X_train, y_train), SpectrogramDataset(X_dev, y_dev)

    def test_train_one_epoch_step(self, small_datasets):
        """train_one_epoch should update weights and decrease loss."""
        train_ds, _ = small_datasets
        loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
        model = SpectrogramCNN(conv_channels=[16, 32], dense_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        loss, acc = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
        assert loss > 0.0
        assert 0.0 <= acc <= 1.0

    def test_evaluate_epoch_metrics(self, small_datasets):
        """evaluate_epoch should compute loss and valid classification metrics."""
        _, dev_ds = small_datasets
        loader = torch.utils.data.DataLoader(dev_ds, batch_size=8, shuffle=False)
        model = SpectrogramCNN(conv_channels=[16, 32], dense_dim=32)
        criterion = nn.CrossEntropyLoss()

        val_loss, metrics = evaluate_epoch(model, loader, criterion, torch.device("cpu"))
        assert val_loss > 0.0
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

    def test_evaluation_partition_guardrail_raises(self, small_datasets):
        """train_model must raise ValueError if 'eval' partition is passed as validation split."""
        train_ds, dev_ds = small_datasets
        model = SpectrogramCNN(conv_channels=[16, 32])

        with pytest.raises(ValueError, match="CRITICAL PROTOCOL VIOLATION"):
            train_model(
                model=model,
                train_dataset=train_ds,
                val_dataset=dev_ds,
                val_partition_name="eval",  # Forbidden: eval cannot be used for checkpoint selection
            )

    def test_train_model_saves_checkpoint_with_metadata(self, small_datasets, tmp_path):
        """train_model should save checkpoint containing model state and preprocessing config."""
        train_ds, dev_ds = small_datasets
        model = SpectrogramCNN(conv_channels=[16, 32], dense_dim=32)
        save_file = str(tmp_path / "test_spec_cnn.pt")

        prep_config = {"audio": {"sample_rate": 16000}, "features": {"mel": {"n_mels": 64}}}

        result = train_model(
            model=model,
            train_dataset=train_ds,
            val_dataset=dev_ds,
            epochs=2,
            batch_size=8,
            save_path=save_file,
            preprocessing_config=prep_config,
            val_partition_name="dev",
        )

        assert Path(save_file).exists()
        ckpt = torch.load(save_file, map_location="cpu")
        assert "model_state_dict" in ckpt
        assert "val_loss" in ckpt
        assert "model_config" in ckpt
        assert ckpt["preprocessing_config"]["features"]["mel"]["n_mels"] == 64
        assert result["best_epoch"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. SpecCNNPredictor Inference Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSpecCNNPredictor:
    """Tests for SpecCNNPredictor single prediction interface."""

    @pytest.fixture
    def checkpoint_file(self, tmp_path):
        """Create and save a dummy trained checkpoint."""
        model = SpectrogramCNN(conv_channels=[16, 32], dense_dim=32)
        save_file = str(tmp_path / "cnn_ckpt.pt")

        payload = {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
            "preprocessing_config": {
                "audio": {"sample_rate": 16000, "duration_sec": 4.0, "target_samples": 64000},
                "features": {"mel": {"n_mels": 64, "n_fft": 1024, "hop_length": 160}},
            },
        }
        torch.save(payload, save_file)
        return save_file

    def test_predictor_from_checkpoint(self, checkpoint_file):
        """Predictor should reload model and parameters from checkpoint."""
        predictor = SpecCNNPredictor.from_checkpoint(checkpoint_file, model_name="spec_cnn")
        assert predictor.model_name == "spec_cnn"
        assert predictor.n_mels == 64

    def test_predict_spectrogram_output_structure(self, checkpoint_file):
        """predict_spectrogram must return standard dictionary with calibrated probabilities."""
        predictor = SpecCNNPredictor.from_checkpoint(checkpoint_file)
        spec = torch.randn(1, 1, 64, 401)
        res = predictor.predict_spectrogram(spec)

        assert "prediction" in res
        assert res["prediction"] in (0, 1)
        assert res["label"] in ("bonafide", "spoof")
        assert 0.0 <= res["probability_spoof"] <= 1.0
        assert 0.0 <= res["probability_bonafide"] <= 1.0
        assert abs(res["probability_spoof"] + res["probability_bonafide"] - 1.0) < 1e-5
        assert res["confidence"] == max(res["probability_spoof"], res["probability_bonafide"])
        assert res["model_name"] == "spec_cnn"

    def test_predict_audio_waveform(self, checkpoint_file):
        """predict_audio should accept 1D audio waveform and perform end-to-end inference."""
        predictor = SpecCNNPredictor.from_checkpoint(checkpoint_file)
        wav = generate_sine_tone(duration_sec=2.0, sample_rate=16000, frequency=440.0)

        res = predictor.predict_audio(wav, sample_rate=16000)
        assert res["prediction"] in (0, 1)
        assert res["label"] in ("bonafide", "spoof")

    def test_inference_determinism_on_same_input(self, checkpoint_file):
        """Consecutive predictions on identical audio must return identical probabilities."""
        predictor = SpecCNNPredictor.from_checkpoint(checkpoint_file)
        wav = generate_sine_tone(duration_sec=3.0, sample_rate=16000, frequency=300.0)

        res1 = predictor.predict_audio(wav, sample_rate=16000)
        res2 = predictor.predict_audio(wav, sample_rate=16000)

        assert res1["prediction"] == res2["prediction"]
        assert abs(res1["probability_spoof"] - res2["probability_spoof"]) < 1e-6
        assert abs(res1["confidence"] - res2["confidence"]) < 1e-6

