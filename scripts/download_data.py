"""Data acquisition and preparation utility for VoxGuard.

Supports:
  1. --subset fixtures: Deterministic synthetic audio fixtures (offline CPU development).
  2. --subset mini: Curated mini development subset for rapid experimentation.
  3. --subset cd_add: External evaluation benchmark (held-out test only, trust_remote_code=False).
  4. --subset full: Full ASVspoof 2019 LA partition acquisition instructions and links.

Usage:
    python scripts/download_data.py --subset mini --samples-per-class 500
    python scripts/download_data.py --subset cd_add --split test
    python scripts/download_data.py --subset full --output-dir data/raw/asvspoof2019_la
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.utils.seed import set_seed


def prepare_fixtures_or_mini(output_dir: str, manifest_dir: str, samples_per_class: int = 500, seed: int = 42):
    """Generate reproducible development fixtures and manifests."""
    print("=" * 70)
    print(" Preparing VoxGuard Synthetic Development Fixtures & Manifests")
    print(" [POLICY] Generated audio consists of synthetic sine/noise fixtures for software testing.")
    print(" [POLICY] Does NOT constitute real speech or deepfake detection data.")
    print("=" * 70)
    set_seed(seed)

    from src.data.test_fixtures import create_test_fixtures
    from src.data.manifest import build_manifests, create_quick_subset, load_manifest_csv, records_to_csv

    fixtures = create_test_fixtures(
        output_dir=output_dir,
        n_bonafide=min(samples_per_class, 50),
        n_spoof=min(samples_per_class * 2, 100),
        n_speakers=6,
        seed=seed,
    )
    print(f"  [OK] Generated {fixtures['total_audio_files']} audio files across 3 partitions.")
    print(f"  - Train speakers: {fixtures['train_speakers']}")
    print(f"  - Dev speakers:   {fixtures['dev_speakers']}")
    print(f"  - Eval speakers:  {fixtures['eval_speakers']}")

    proto_dir = str(Path(output_dir) / "protocols")
    audit = build_manifests(proto_dir, manifest_dir)
    print(f"  [OK] Manifests generated in {manifest_dir}:")
    for part, count in audit["partition_counts"].items():
        # Tag manifest with dataset_mode
        p_csv = Path(manifest_dir) / f"asvspoof_{part}.csv"
        records = load_manifest_csv(str(p_csv))
        for r in records:
            r["dataset_mode"] = "synthetic_fixture"
        records_to_csv(records, str(p_csv), columns=["utterance_id", "speaker_id", "system_id", "label_str", "label", "partition", "dataset_mode"])
        print(f"    - asvspoof_{part}.csv: {count} utterances (mode: synthetic_fixture)")

    # Create quick subsets
    for part in ["train", "dev", "eval"]:
        src = str(Path(manifest_dir) / f"asvspoof_{part}.csv")
        dst = str(Path(manifest_dir) / f"asvspoof_{part}_mini.csv")
        create_quick_subset(src, dst, samples_per_class=samples_per_class, seed=seed)
        # Ensure subset carries dataset_mode
        sub_records = load_manifest_csv(dst)
        for r in sub_records:
            r["dataset_mode"] = "synthetic_fixture"
        records_to_csv(sub_records, dst, columns=["utterance_id", "speaker_id", "system_id", "label_str", "label", "partition", "dataset_mode"])
        print(f"    - asvspoof_{part}_mini.csv created (mode: synthetic_fixture).")

    print("\nDevelopment fixture data preparation complete.")


def prepare_cd_add(output_path: str = "data/manifests/cd_add_test.csv", split: str = "test", max_samples: int = 2000):
    """Prepare held-out external CD-ADD test manifest."""
    print("=" * 70)
    print(" Preparing External CD-ADD Held-Out Test Manifest")
    print("=" * 70)
    print("  [POLICY] CD-ADD is STRICTLY external evaluation data.")
    print("  [POLICY] Never used for training, normalizer fitting, or threshold tuning.")
    print("  [SECURITY] trust_remote_code=False enforced.")

    from src.data.cd_add_loader import build_cd_add_manifest

    try:
        summary = build_cd_add_manifest(
            output_path=output_path,
            split=split,
            max_samples=max_samples,
            trust_remote_code=False,
        )
        print(f"  [OK] Successfully prepared CD-ADD test manifest with {summary['total_records']} samples.")
        print(f"  - Output: {summary['output_path']}")
        print(f"  - Class distribution: {summary['class_distribution']}")
    except Exception as e:
        print(f"  [NOTE] Remote dataset download could not complete: {e}")
        print("  To load CD-ADD on Google Colab or an unsandboxed machine:")
        print("    pip install datasets huggingface_hub")
        print("    python scripts/download_data.py --subset cd_add --split test")


def print_full_dataset_instructions(output_dir: str):
    """Print verified instructions and links for downloading the full ASVspoof 2019 LA partition."""
    print("=" * 70)
    print(" Full ASVspoof 2019 Logical Access (LA) Ingestion Instructions")
    print("=" * 70)
    print(f"Target Directory: {output_dir}\n")
    print("The complete ASVspoof 2019 LA dataset is ~15-20 GB and distributed via Edinburgh DataShare:")
    print("  URL: https://datashare.ed.ac.uk/handle/10283/3336\n")
    print("Direct Archive Files:")
    print("  1. LA.zip (contains train, dev, and eval audio in .flac format)")
    print("     Download Link: https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip\n")
    print("Protocol Files:")
    print("  Contained within LA.zip inside directory LA/ASVspoof2019_LA_cm_protocols/:")
    print("    - ASVspoof2019.LA.cm.train.trn.txt")
    print("    - ASVspoof2019.LA.cm.dev.trl.txt")
    print("    - ASVspoof2019.LA.cm.eval.trl.txt\n")
    print("Automated Unpacking Command (PowerShell / Bash):")
    print(f"  mkdir -p {output_dir}")
    print(f"  # After downloading LA.zip to {output_dir}:")
    print(f"  unzip LA.zip -d {output_dir}\n")
    print("After unpacking, generate manifests with:")
    print(f"  python scripts/run_eda.py --config configs/default.yaml\n")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="VoxGuard Dataset Preparation")
    parser.add_argument("--subset", choices=["mini", "fixtures", "cd_add", "full"], default="mini",
                        help="Data preparation target")
    parser.add_argument("--samples-per-class", type=int, default=500,
                        help="Sample count per class for mini/fixtures")
    parser.add_argument("--split", default="test", choices=["test"],
                        help="Split for CD-ADD (must be 'test')")
    parser.add_argument("--output-dir", default=None,
                        help="Target output directory")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to default configuration")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)

    if args.subset in ("mini", "fixtures"):
        out_dir = args.output_dir or cfg["paths"]["raw_asvspoof_dir"]
        manifest_dir = cfg["paths"]["manifest_dir"]
        prepare_fixtures_or_mini(out_dir, manifest_dir, samples_per_class=args.samples_per_class, seed=seed)
    elif args.subset == "cd_add":
        out_manifest = str(Path(cfg["paths"]["manifest_dir"]) / "cd_add_test.csv")
        prepare_cd_add(output_path=out_manifest, split=args.split)
    elif args.subset == "full":
        out_dir = args.output_dir or cfg["paths"]["raw_asvspoof_dir"]
        print_full_dataset_instructions(out_dir)


if __name__ == "__main__":
    main()

