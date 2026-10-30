"""Dataset audit and exploratory data analysis script.

Generates:
  - Dataset audit summary (partition counts, class balance, speaker info)
  - Class distribution bar plots per partition
  - Duration distribution histograms per partition and label
  - System/algorithm distribution plots

Usage:
    python scripts/run_eda.py --config configs/default.yaml
    python scripts/run_eda.py --fixtures    # Use synthetic test fixtures
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure repository root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.protocol_parser import (
    get_class_distribution,
    get_system_distribution,
    load_all_partitions,
    validate_no_duplicate_utterances,
    validate_speaker_disjointness,
)
from src.data.manifest import build_manifests, create_quick_subset


def plot_class_distribution(class_dist, figures_dir):
    """Plot class distribution per partition as grouped bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plots.")
        return

    partitions = sorted(class_dist.keys())
    labels = ["bonafide", "spoof"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(partitions))
    width = 0.35

    bonafide_counts = [class_dist[p].get("bonafide", 0) for p in partitions]
    spoof_counts = [class_dist[p].get("spoof", 0) for p in partitions]

    bars1 = ax.bar(x - width / 2, bonafide_counts, width, label="Bonafide (Genuine)",
                   color="#2ecc71", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, spoof_counts, width, label="Spoof (Synthetic)",
                   color="#e74c3c", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Partition", fontsize=12)
    ax.set_ylabel("Number of Utterances", fontsize=12)
    ax.set_title("ASVspoof 2019 LA — Class Distribution per Partition", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in partitions])
    ax.legend()

    # Add count labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{int(height):,}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{int(height):,}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = Path(figures_dir) / "class_distribution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_system_distribution(system_dist, figures_dir):
    """Plot spoofing system distribution per partition."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for partition, systems in sorted(system_dist.items()):
        sys_ids = sorted(systems.keys())
        counts = [systems[s] for s in sys_ids]

        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ["#2ecc71" if s == "-" else "#3498db" for s in sys_ids]
        ax.bar(sys_ids, counts, color=colors, edgecolor="black", linewidth=0.5)

        ax.set_xlabel("System / Algorithm ID", fontsize=12)
        ax.set_ylabel("Number of Utterances", fontsize=12)
        ax.set_title(f"ASVspoof 2019 LA — System Distribution ({partition.capitalize()})",
                     fontsize=14)

        for i, (sid, cnt) in enumerate(zip(sys_ids, counts)):
            ax.annotate(f"{cnt:,}",
                        xy=(i, cnt), xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        out = Path(figures_dir) / f"system_distribution_{partition}.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        print(f"  Saved: {out}")


def plot_duration_distribution(records, audio_base_dir, figures_dir, max_files=500):
    """Plot duration distribution per partition and label (from audio files)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    audio_base = Path(audio_base_dir) if audio_base_dir else None
    if audio_base is None or not audio_base.exists():
        print("  [WARN] Audio directory not found, skipping duration plots.")
        return

    from src.data.preprocessor import load_audio_file

    durations = {"train": {"bonafide": [], "spoof": []},
                 "dev": {"bonafide": [], "spoof": []},
                 "eval": {"bonafide": [], "spoof": []}}

    sampled = 0
    rng = np.random.RandomState(42)
    indices = rng.choice(len(records), size=min(max_files, len(records)), replace=False)

    for idx in indices:
        rec = records[idx]
        partition = rec["partition"]
        label = rec["label_str"]
        uid = rec["utterance_id"]

        # Try to find the audio file
        for ext in [".flac", ".wav"]:
            candidate = audio_base / partition / f"{uid}{ext}"
            if not candidate.exists():
                candidate = audio_base / partition / "flac" / f"{uid}{ext}"
            if candidate.exists():
                waveform, sr, err = load_audio_file(str(candidate))
                if waveform is not None and sr > 0:
                    dur = len(waveform) / sr
                    if partition in durations:
                        durations[partition][label].append(dur)
                    sampled += 1
                break

    if sampled == 0:
        print("  [WARN] No audio files found, skipping duration plots.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, partition in enumerate(["train", "dev", "eval"]):
        ax = axes[i]
        for label, color in [("bonafide", "#2ecc71"), ("spoof", "#e74c3c")]:
            durs = durations[partition][label]
            if durs:
                ax.hist(durs, bins=30, alpha=0.6, color=color,
                        label=f"{label} (n={len(durs)})", edgecolor="black",
                        linewidth=0.3)
        ax.set_xlabel("Duration (seconds)")
        ax.set_title(f"{partition.capitalize()}")
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Count")
    fig.suptitle("ASVspoof 2019 LA — Audio Duration Distribution", fontsize=14)
    plt.tight_layout()

    out = Path(figures_dir) / "duration_distribution.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def run_audit(config_path="configs/default.yaml", mode="real", use_fixtures=False):
    """Run the full dataset audit and EDA pipeline.

    Args:
        config_path: Path to configuration YAML.
        mode: Dataset mode ('fixture' or 'real').
        use_fixtures: Deprecated alias for mode='fixture'.
    """
    if use_fixtures:
        mode = "fixture"

    print("=" * 70)
    print(f"  VoxGuard Dataset Audit & Exploratory Data Analysis (Mode: {mode.upper()})")
    print("=" * 70)

    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    figures_dir = config["paths"]["figures_dir"]
    manifest_dir = config["paths"]["manifest_dir"]

    if mode == "fixture":
        print("\n[1] Preparing synthetic test fixtures harness...")
        from src.data.test_fixtures import create_test_fixtures
        fixtures = create_test_fixtures(output_dir="tests/fixtures")
        protocol_dir = "tests/fixtures/protocols"
        audio_base_dir = "tests/fixtures/audio"
        dataset_mode = "synthetic_fixture"
        audio_type = "procedural_sine_noise_fixture"
        print(f"  Generated {fixtures['total_audio_files']} synthetic audio fixture files.")
        print(f"  Train speakers: {fixtures['train_speakers']}")
        print(f"  Dev speakers: {fixtures['dev_speakers']}")
    else:
        protocol_dir = config["paths"]["raw_asvspoof_dir"]
        audio_base_dir = config["paths"]["raw_asvspoof_dir"]
        dataset_mode = "real_speech"
        audio_type = "human_speech_recording"
        required_protocol = Path(protocol_dir) / "ASVspoof2019.LA.cm.train.trn.txt"
        if not required_protocol.exists():
            raise FileNotFoundError(
                f"Real speech protocol file missing: {required_protocol}. "
                f"Silent fallback to fixtures is blocked by scientific integrity policy. "
                f"Either acquire real ASVspoof data (python scripts/download_data.py --subset full) "
                f"or run explicitly in fixture mode (python scripts/run_eda.py --mode fixture)."
            )

    # Step 1: Parse protocols and build manifests
    print(f"\n[2] Parsing ASVspoof 2019 LA protocol files from: {protocol_dir}")
    try:
        all_records = load_all_partitions(protocol_dir)
        print(f"  Total records parsed: {len(all_records)}")
    except FileNotFoundError as e:
        print(f"  [ERROR] Protocol files not found: {e}")
        print("  Run with --fixtures flag to use synthetic test data.")
        return

    # Step 2: Class distribution
    print("\n[3] Class Distribution:")
    class_dist = get_class_distribution(all_records)
    for partition, dist in sorted(class_dist.items()):
        total = sum(dist.values())
        bonafide = dist.get("bonafide", 0)
        spoof = dist.get("spoof", 0)
        ratio = spoof / bonafide if bonafide > 0 else float("inf")
        print(f"  {partition:>6s}: bonafide={bonafide:>6,}  spoof={spoof:>6,}  "
              f"total={total:>6,}  ratio(spoof:bonafide)={ratio:.1f}:1")

    # Step 3: System distribution
    print("\n[4] Spoofing System Distribution:")
    system_dist = get_system_distribution(all_records)
    for partition, systems in sorted(system_dist.items()):
        print(f"  {partition}: {dict(sorted(systems.items()))}")

    # Step 4: Speaker disjointness check
    print("\n[5] Speaker Disjointness Validation:")
    speaker_check = validate_speaker_disjointness(all_records)
    print(f"  Speakers per partition: {speaker_check['partition_speaker_counts']}")
    if speaker_check["is_disjoint"]:
        print("  [PASS] All partitions are speaker-disjoint.")
    else:
        print(f"  [FAIL] Overlapping speakers found: {speaker_check['overlapping_speakers']}")

    # Step 5: Duplicate utterance check
    print("\n[6] Duplicate Utterance Detection:")
    dup_check = validate_no_duplicate_utterances(all_records)
    print(f"  Total utterances: {dup_check['total_utterances']}")
    print(f"  Unique utterances: {dup_check['unique_utterances']}")
    if dup_check["has_cross_partition_leaks"]:
        print(f"  [FAIL] Cross-partition duplicates: {dup_check['cross_partition_duplicates']}")
    else:
        print("  [PASS] No cross-partition duplicate utterances.")
    if dup_check["has_within_partition_duplicates"]:
        print(f"  [WARN] Within-partition duplicates: {dup_check['within_partition_duplicates']}")
    else:
        print("  [PASS] No within-partition duplicate utterances.")

    # Step 6: Save manifests
    print(f"\n[7] Writing manifest CSVs to: {manifest_dir}")
    from src.data.manifest import records_to_csv
    partitioned = {}
    for rec in all_records:
        p = rec["partition"]
        if p not in partitioned:
            partitioned[p] = []
        partitioned[p].append(rec)

    for partition, records in partitioned.items():
        for r in records:
            r["dataset_mode"] = dataset_mode
        out_path = str(Path(manifest_dir) / f"asvspoof_{partition}.csv")
        n = records_to_csv(
            records,
            out_path,
            columns=["utterance_id", "speaker_id", "system_id", "label_str", "label", "partition", "dataset_mode"],
        )
        print(f"  {partition}: {n} records -> {out_path}")

    # Step 7: Generate plots
    print(f"\n[8] Generating EDA plots to: {figures_dir}")
    plot_class_distribution(class_dist, figures_dir)
    plot_system_distribution(system_dist, figures_dir)
    plot_duration_distribution(all_records, audio_base_dir, figures_dir)

    # Step 8: Quick subset
    print("\n[9] Creating quick-experiment subset:")
    for partition in partitioned:
        manifest_path = str(Path(manifest_dir) / f"asvspoof_{partition}.csv")
        subset_path = str(Path(manifest_dir) / f"asvspoof_{partition}_mini.csv")
        try:
            summary = create_quick_subset(manifest_path, subset_path,
                                          samples_per_class=500, seed=42)
            print(f"  {partition} subset: {summary['total_subset_samples']} samples "
                  f"({summary['class_counts']})")
        except Exception as e:
            print(f"  {partition} subset: [WARN] {e}")

    # Audit summary
    print("\n" + "=" * 70)
    print("  AUDIT SUMMARY")
    print("=" * 70)
    audit = {
        "dataset_mode": dataset_mode,
        "audio_type": audio_type,
        "is_synthetic_fixture": (dataset_mode == "synthetic_fixture"),
        "total_records": len(all_records),
        "class_distribution": class_dist,
        "speaker_disjoint": speaker_check["is_disjoint"],
        "no_cross_partition_duplicates": not dup_check["has_cross_partition_leaks"],
        "partition_counts": {p: len(r) for p, r in partitioned.items()},
    }
    print(json.dumps(audit, indent=2, default=str))

    # Save audit report
    audit_path = Path(config["paths"]["results_dir"]) / "dataset_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"\n  Audit report saved to: {audit_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoxGuard Dataset Audit & EDA")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to configuration YAML file")
    parser.add_argument("--mode", choices=["fixture", "real"], default="real",
                        help="Operating mode: 'fixture' for synthetic tests, 'real' for real speech")
    parser.add_argument("--fixtures", action="store_true",
                        help="Use synthetic test fixtures instead of real data (alias for --mode fixture)")
    args = parser.parse_args()
    selected_mode = "fixture" if args.fixtures else args.mode
    run_audit(args.config, mode=selected_mode)

