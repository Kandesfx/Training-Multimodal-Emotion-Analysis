"""Merge emotion labels into aligned_50.pkl and unaligned_50.pkl.

Run this script on Google Colab after download_emotion_labels.py.

Usage:
    python scripts/merge_emotions_to_pkl.py \
        --emotion-labels /content/data/emotion_labels.pkl \
        --aligned-pkl /content/data/MSA-Dataset/aligned_50.pkl \
        --unaligned-pkl /content/data/MSA-Dataset/unaligned_50.pkl

This adds an 'emotion_labels' key to each split:
    data[split]['emotion_labels']  # shape: (N, 6) float32
    # Order: [happy, sad, angry, surprise, disgust, fear]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


EMOTION_COLS = ["happy", "sad", "angry", "surprise", "disgust", "fear"]


def load_emotion_labels(path: str) -> dict:
    """Load emotion labels dict from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def try_match_id(sample_id: str, emotion_dict: dict) -> dict | None:
    """Try to match a sample ID to emotion labels.

    MOSEI sample IDs in the pkl use format: '{video_id}$_${segment_number}'
    The CMU SDK may use different segment numbering.
    Try exact match first, then fuzzy match by video_id + nearby segments.
    """
    # Exact match
    if sample_id in emotion_dict:
        return emotion_dict[sample_id]

    # Parse video_id and segment
    parts = sample_id.split("$_$")
    if len(parts) != 2:
        return None

    video_id, seg_str = parts

    # Try with segment number as-is
    for key in emotion_dict:
        if key.startswith(f"{video_id}$_$"):
            key_seg = key.split("$_$")[1]
            if key_seg == seg_str:
                return emotion_dict[key]

    return None


def merge_into_pkl(pkl_path: str, emotion_dict: dict, output_path: str | None = None) -> None:
    """Merge emotion labels into a MOSEI pkl file."""
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        print(f"  SKIP: {pkl_path} not found")
        return

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    out_path = Path(output_path) if output_path else pkl_path

    for split in ["train", "valid", "test"]:
        if split not in data:
            continue

        split_data = data[split]
        sample_ids = split_data.get("id", [])
        n = len(sample_ids)

        emotion_labels = np.zeros((n, 6), dtype=np.float32)
        matched_mask = np.zeros(n, dtype=bool)
        matched = 0
        unmatched_ids = []

        for i, sid in enumerate(sample_ids):
            emo = try_match_id(sid, emotion_dict)
            if emo is not None:
                emotion_labels[i] = [emo[col] for col in EMOTION_COLS]
                matched_mask[i] = True
                matched += 1
            else:
                unmatched_ids.append(sid)

        coverage = matched / n * 100 if n > 0 else 0
        split_data["emotion_labels"] = emotion_labels
        split_data["emotion_matched_mask"] = matched_mask

        print(f"  {split}: {matched}/{n} matched ({coverage:.1f}%)")
        if unmatched_ids and len(unmatched_ids) <= 10:
            print(f"    Unmatched: {unmatched_ids}")
        elif unmatched_ids:
            print(f"    First 5 unmatched: {unmatched_ids[:5]}")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"  Saved to: {out_path}")

    # Verify
    with open(out_path, "rb") as f:
        verify = pickle.load(f)
    for split in ["train", "valid", "test"]:
        emo = verify[split].get("emotion_labels")
        if emo is not None:
            print(f"  Verify {split}: emotion_labels shape={emo.shape}, "
                  f"non-zero rows={np.any(emo > 0, axis=1).sum()}")


def main():
    parser = argparse.ArgumentParser(description="Merge emotion labels into MOSEI pkl files")
    parser.add_argument("--emotion-labels", type=str, required=True,
                        help="Path to emotion_labels.pkl")
    parser.add_argument("--aligned-pkl", type=str,
                        default="/content/data/MSA-Dataset/aligned_50.pkl")
    parser.add_argument("--unaligned-pkl", type=str,
                        default="/content/data/MSA-Dataset/unaligned_50.pkl")
    parser.add_argument("--inplace", action="store_true", default=True,
                        help="Overwrite original pkl files (default: True)")
    args = parser.parse_args()

    emotion_dict = load_emotion_labels(args.emotion_labels)
    print(f"Loaded {len(emotion_dict)} emotion label entries\n")

    print(f"=== Merging into aligned pkl ===")
    merge_into_pkl(args.aligned_pkl, emotion_dict)

    print(f"\n=== Merging into unaligned pkl ===")
    merge_into_pkl(args.unaligned_pkl, emotion_dict)

    print("\nDone! Emotion labels merged. Upload updated pkl files to GCS if needed:")
    print("  gsutil cp /content/data/MSA-Dataset/aligned_50.pkl gs://mer-data-bucket-kandesfx/data/MSA-Dataset/")
    print("  gsutil cp /content/data/MSA-Dataset/unaligned_50.pkl gs://mer-data-bucket-kandesfx/data/MSA-Dataset/")


if __name__ == "__main__":
    main()
