"""Prepare and partition a controlled 2,000-sample subset of MLAAD-tiny for VoxGuard.

Specifications:
- Total samples: 2,000 (1,000 Genuine Human, 1,000 Synthetic/AI-Generated)
- Language: English (controlled)
- Partition Split:
    * Train: 1,400 samples (700 genuine, 700 synthetic)
    * Dev: 300 samples (150 genuine, 150 synthetic)
    * Test: 300 samples (150 genuine, 150 synthetic)
- Disjoint Speaker Partitioning (Zero Speaker Leakage, Verified by M-AILABS metadata):
    * Train: 700 samples from wives_and_daughters (Speaker: elizabeth_klett, female)
    * Dev:   150 samples from midnight_passenger (Speaker: mary_ann, female)
    * Test:  150 samples from poisoned_pen (Speaker: elliot_miller, male)
- Controlled Generator Partitioning:
    * Train: 10 modern TTS systems (70 samples each = 700):
        Edge-TTS, FishTTS, Chatterbox, DeepGram, Indri-TTS-0.1,
        Llasa-1B-Multilingual, ZipVoice, Openaudio-S1-Mini, Veena, griffin_lim
    * Dev: 2 modern TTS systems (75 samples each = 150):
        Higgs-Audio-V2, Index-TTS-1.5
    * Test: 75 Seen generator samples (15 each from 5 train generators: Edge-TTS, FishTTS, Chatterbox, DeepGram, ZipVoice) +
            75 Unseen generator samples (25 each: Cartesia.ai (Sonic-3), Mars5, MatchaTTS)
- Manifests saved to:
    * data/manifests/mlaad_train.csv & .json
    * data/manifests/mlaad_dev.csv & .json
    * data/manifests/mlaad_test.csv & .json
    * data/manifests/mlaad_complete_manifest.csv
"""

import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from huggingface_hub import hf_hub_download, list_repo_files
import soundfile as sf

REPO_ID = "mueller91/MLAAD-tiny"
DEST_DIR = os.path.abspath("data/raw/mlaad_subset")
MANIFEST_DIR = os.path.abspath("data/manifests")


def download_and_verify(item: Dict[str, object]) -> Dict[str, object]:
    """Download a single file from HF hub, decode with soundfile, and record metadata."""
    hf_path = item["hf_path"]
    try:
        local_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=hf_path)
        info = sf.info(local_path)
        item["local_path"] = local_path
        item["duration_sec"] = round(info.duration, 4)
        item["sample_rate"] = info.samplerate
        item["channels"] = info.channels
        item["valid"] = True
        item["error"] = ""
    except Exception as e:
        item["local_path"] = ""
        item["duration_sec"] = 0.0
        item["sample_rate"] = 0
        item["channels"] = 0
        item["valid"] = False
        item["error"] = str(e)
    return item


def save_manifests(partition_name: str, records: List[Dict[str, object]]):
    """Save partition records as CSV and JSON."""
    csv_path = os.path.join(MANIFEST_DIR, f"mlaad_{partition_name}.csv")
    json_path = os.path.join(MANIFEST_DIR, f"mlaad_{partition_name}.json")

    fieldnames = [
        "utterance_id",
        "label",
        "label_str",
        "speaker_id",
        "speaker_name",
        "speaker_gender",
        "system_id",
        "generator_status",
        "partition",
        "duration_sec",
        "sample_rate",
        "channels",
        "hf_path",
        "local_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {partition_name} manifest: {csv_path} ({len(records)} records)")


def main():
    start_time = time.time()
    print("=" * 80)
    print("PREPARING CONTROLLED MLAAD-TINY EXPERIMENTAL SUBSET")
    print("=" * 80)

    os.makedirs(DEST_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    print("Fetching repository file listing from Hugging Face...")
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    print(f"Total files in repository: {len(all_files):,}")

    # Strictly filter for .wav audio files only
    orig_en = sorted([f for f in all_files if f.startswith("original/en/") and f.endswith(".wav")])
    fake_en = sorted([f for f in all_files if f.startswith("fake/en/") and f.endswith(".wav")])

    print(f"Available English WAV files: {len(orig_en):,} Genuine, {len(fake_en):,} Fake")

    items_to_download: List[Dict[str, object]] = []

    # -------------------------------------------------------------------------
    # 1. Genuine Speech Partitions (Zero Speaker Leakage, Verified Metadata)
    # -------------------------------------------------------------------------
    train_orig_files = [f for f in orig_en if "wives_and_daughters" in f][:700]
    dev_orig_files = [f for f in orig_en if "midnight_passenger" in f][:150]
    test_orig_files = [f for f in orig_en if "poisoned_pen" in f][:150]

    assert len(train_orig_files) == 700, f"Expected 700 train genuine, got {len(train_orig_files)}"
    assert len(dev_orig_files) == 150, f"Expected 150 dev genuine, got {len(dev_orig_files)}"
    assert len(test_orig_files) == 150, f"Expected 150 test genuine, got {len(test_orig_files)}"

    for idx, f in enumerate(train_orig_files):
        items_to_download.append({
            "utterance_id": f"MLAAD_TR_GEN_{idx+1:04d}",
            "hf_path": f,
            "label": 0,
            "label_str": "bonafide",
            "speaker_id": "spk_elizabeth_klett",
            "speaker_name": "Elizabeth Klett",
            "speaker_gender": "female",
            "system_id": "human_bonafide",
            "generator_status": "human",
            "partition": "train",
        })

    for idx, f in enumerate(dev_orig_files):
        items_to_download.append({
            "utterance_id": f"MLAAD_DEV_GEN_{idx+1:04d}",
            "hf_path": f,
            "label": 0,
            "label_str": "bonafide",
            "speaker_id": "spk_mary_ann",
            "speaker_name": "Mary Ann",
            "speaker_gender": "female",
            "system_id": "human_bonafide",
            "generator_status": "human",
            "partition": "dev",
        })

    for idx, f in enumerate(test_orig_files):
        items_to_download.append({
            "utterance_id": f"MLAAD_TE_GEN_{idx+1:04d}",
            "hf_path": f,
            "label": 0,
            "label_str": "bonafide",
            "speaker_id": "spk_elliot_miller",
            "speaker_name": "Elliot Miller",
            "speaker_gender": "male",
            "system_id": "human_bonafide",
            "generator_status": "human",
            "partition": "test",
        })

    # -------------------------------------------------------------------------
    # 2. Synthetic Speech Partitions
    # -------------------------------------------------------------------------
    train_gens = [
        "Edge-TTS", "FishTTS", "Chatterbox", "DeepGram", "Indri-TTS-0.1",
        "Llasa-1B-Multilingual", "ZipVoice", "Openaudio-S1-Mini", "Veena", "griffin_lim"
    ]
    train_fake_idx = 0
    for gen in train_gens:
        gen_files = [f for f in fake_en if f.startswith(f"fake/en/{gen}/")][:70]
        assert len(gen_files) == 70, f"Not enough files for {gen}: {len(gen_files)}"
        for f in gen_files:
            train_fake_idx += 1
            items_to_download.append({
                "utterance_id": f"MLAAD_TR_SPOOF_{train_fake_idx:04d}",
                "hf_path": f,
                "label": 1,
                "label_str": "spoof",
                "speaker_id": "synthetic",
                "speaker_name": "Synthetic",
                "speaker_gender": "synthetic",
                "system_id": gen,
                "generator_status": "seen_in_train",
                "partition": "train",
            })
    assert train_fake_idx == 700

    # Dev Synthetic: 2 generators @ 75 samples = 150
    dev_gens = ["Higgs-Audio-V2", "Index-TTS-1.5"]
    dev_fake_idx = 0
    for gen in dev_gens:
        gen_files = [f for f in fake_en if f.startswith(f"fake/en/{gen}/")][:75]
        assert len(gen_files) == 75, f"Not enough files for {gen}: {len(gen_files)}"
        for f in gen_files:
            dev_fake_idx += 1
            items_to_download.append({
                "utterance_id": f"MLAAD_DEV_SPOOF_{dev_fake_idx:04d}",
                "hf_path": f,
                "label": 1,
                "label_str": "spoof",
                "speaker_id": "synthetic",
                "speaker_name": "Synthetic",
                "speaker_gender": "synthetic",
                "system_id": gen,
                "generator_status": "unseen_in_train",
                "partition": "dev",
            })
    assert dev_fake_idx == 150

    # Test Synthetic: 75 Seen (15 each from 5 train generators, files 70:85) + 75 Unseen (25 each: Cartesia, Mars5, MatchaTTS)
    seen_test_gens = ["Edge-TTS", "FishTTS", "Chatterbox", "DeepGram", "ZipVoice"]
    test_fake_idx = 0
    for gen in seen_test_gens:
        gen_files = [f for f in fake_en if f.startswith(f"fake/en/{gen}/")][70:85]
        assert len(gen_files) == 15, f"Not enough seen test files for {gen}: {len(gen_files)}"
        for f in gen_files:
            test_fake_idx += 1
            items_to_download.append({
                "utterance_id": f"MLAAD_TE_SPOOF_{test_fake_idx:04d}",
                "hf_path": f,
                "label": 1,
                "label_str": "spoof",
                "speaker_id": "synthetic",
                "speaker_name": "Synthetic",
                "speaker_gender": "synthetic",
                "system_id": gen,
                "generator_status": "seen_generator",
                "partition": "test",
            })

    unseen_test_gens = ["Cartesia.ai (Sonic-3)", "Mars5", "MatchaTTS"]
    for gen in unseen_test_gens:
        gen_files = [f for f in fake_en if f.startswith(f"fake/en/{gen}/")][:25]
        assert len(gen_files) == 25, f"Not enough unseen test files for {gen}: {len(gen_files)}"
        for f in gen_files:
            test_fake_idx += 1
            items_to_download.append({
                "utterance_id": f"MLAAD_TE_SPOOF_{test_fake_idx:04d}",
                "hf_path": f,
                "label": 1,
                "label_str": "spoof",
                "speaker_id": "synthetic",
                "speaker_name": "Synthetic",
                "speaker_gender": "synthetic",
                "system_id": gen,
                "generator_status": "unseen_generator",
                "partition": "test",
            })
    assert test_fake_idx == 150

    # Total items must be exactly 2,000
    assert len(items_to_download) == 2000, f"Expected 2,000 items, got {len(items_to_download)}"

    # Check partition distribution
    train_items = [x for x in items_to_download if x["partition"] == "train"]
    dev_items = [x for x in items_to_download if x["partition"] == "dev"]
    test_items = [x for x in items_to_download if x["partition"] == "test"]

    print("\n[Target Dataset Split Summary]:")
    print(f"  - Train: {len(train_items)} samples ({len([x for x in train_items if x['label']==0])} Bonafide, {len([x for x in train_items if x['label']==1])} Spoof)")
    print(f"  - Dev:   {len(dev_items)} samples ({len([x for x in dev_items if x['label']==0])} Bonafide, {len([x for x in dev_items if x['label']==1])} Spoof)")
    print(f"  - Test:  {len(test_items)} samples ({len([x for x in test_items if x['label']==0])} Bonafide, {len([x for x in test_items if x['label']==1])} Spoof)")
    print(f"  - Total: {len(items_to_download)} samples")

    # -------------------------------------------------------------------------
    # 3. Download and Verify with 16 Workers
    # -------------------------------------------------------------------------
    print(f"\nStarting concurrent download and verification of 2,000 files (16 workers)...")
    verified_records: List[Dict[str, object]] = []
    completed_count = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_and_verify, item): item for item in items_to_download}
        for future in as_completed(futures):
            res = future.result()
            verified_records.append(res)
            completed_count += 1
            if completed_count % 400 == 0 or completed_count == 2000:
                elapsed = time.time() - t0
                rate = completed_count / elapsed if elapsed > 0 else 0
                print(f"  Progress: {completed_count}/2000 files verified ({rate:.1f} files/sec, {elapsed:.1f}s elapsed)")

    # Sort records to maintain deterministic order
    verified_records.sort(key=lambda x: x["utterance_id"])

    # Verify no decoding failures
    failed = [r for r in verified_records if not r["valid"]]
    if failed:
        print(f"\n[WARNING] {len(failed)} files failed decoding:")
        for r in failed[:5]:
            print(f"  {r['hf_path']}: {r['error']}")
        raise RuntimeError("Decoding verification failed for some samples.")
    else:
        print("\nAll 2,000 files successfully downloaded and verified via soundfile!")

    # -------------------------------------------------------------------------
    # 4. Save Manifests
    # -------------------------------------------------------------------------
    train_recs = [r for r in verified_records if r["partition"] == "train"]
    dev_recs = [r for r in verified_records if r["partition"] == "dev"]
    test_recs = [r for r in verified_records if r["partition"] == "test"]

    save_manifests("train", train_recs)
    save_manifests("dev", dev_recs)
    save_manifests("test", test_recs)

    # Save complete unified manifest
    all_csv_path = os.path.join(MANIFEST_DIR, "mlaad_complete_manifest.csv")
    fieldnames = [
        "utterance_id", "label", "label_str", "speaker_id", "speaker_name", "speaker_gender", "system_id",
        "generator_status", "partition", "duration_sec", "sample_rate",
        "channels", "hf_path", "local_path"
    ]
    with open(all_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in verified_records:
            writer.writerow(r)
    print(f"Saved complete unified manifest: {all_csv_path} ({len(verified_records)} records)")

    total_dur = sum(r["duration_sec"] for r in verified_records)
    print(f"\nSubset Audio Statistics:")
    print(f"  - Total audio duration: {total_dur/3600:.2f} hours ({total_dur:.1f} seconds)")
    print(f"  - Average sample duration: {total_dur/len(verified_records):.2f} seconds")
    print(f"  - Total script execution time: {time.time() - start_time:.2f} seconds")
    print("=" * 80)


if __name__ == "__main__":
    main()
