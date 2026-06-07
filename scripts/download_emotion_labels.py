"""Download CMU-MOSEI emotion labels from CMU MultiComp SDK.

Run this script on Google Colab (needs internet + disk space).

Usage:
    pip install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
    python scripts/download_emotion_labels.py --output /content/data/emotion_labels.pkl

Output: emotion_labels.pkl containing:
    {
        "<video_id>$_$<segment>": {
            "sentiment": float,     # -3 to +3
            "happy": float,         # 0 to 3
            "sad": float,           # 0 to 3
            "angry": float,         # 0 to 3
            "surprise": float,      # 0 to 3
            "disgust": float,       # 0 to 3
            "fear": float,          # 0 to 3
        },
        ...
    }
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


EMOTION_NAMES = ["sentiment", "happy", "sad", "angry", "surprise", "disgust", "fear"]


def download_and_parse(download_dir: str = "/content/cmu_mosei_labels") -> dict:
    """Download CMU-MOSEI labels and parse into a flat dict keyed by sample ID."""
    try:
        from mmsdk import mmdatasdk as md
    except ImportError:
        print("ERROR: mmsdk not installed. Run:")
        print("  pip install git+https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git")
        sys.exit(1)

    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    csd_file = download_path / "CMU_MOSEI_Labels.csd"
    if csd_file.exists():
        print(f"Using cached labels from {csd_file}")
        label_dataset = md.mmdataset({"labels": str(csd_file)})
    else:
        print("Downloading CMU-MOSEI labels...")
        try:
            label_dataset = md.mmdataset(md.cmu_mosei.labels, str(download_path))
        except Exception as e:
            print(f"Download failed: {e}")
            raise

    # Get the computational sequence
    label_key = list(label_dataset.computational_sequences.keys())[0]
    labels_data = label_dataset.computational_sequences[label_key].data

    print(f"Total videos in CMU-MOSEI labels: {len(labels_data)}")

    # Parse: each video has segments with label vectors
    emotion_dict = {}
    n_segments = 0

    for video_id, segments in labels_data.items():
        # segments['features'] shape: (num_segments, 7)
        # segments['intervals'] shape: (num_segments, 2) — start/end times
        features = segments["features"]
        intervals = segments["intervals"]

        for seg_idx in range(len(features)):
            label_vector = features[seg_idx]  # [sentiment, happy, sad, angry, surprise, disgust, fear]

            # Create sample ID matching pkl format: {video_id}$_${segment_index}
            # MOSEI uses segment indices based on intervals
            sample_id = f"{video_id}$_${seg_idx}"

            emotion_dict[sample_id] = {
                name: float(label_vector[i])
                for i, name in enumerate(EMOTION_NAMES)
            }
            n_segments += 1

    print(f"Parsed {n_segments} segments from {len(labels_data)} videos")
    return emotion_dict


def save_labels(emotion_dict: dict, output_path: str) -> None:
    """Save emotion labels to pickle file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(emotion_dict, f)
    print(f"Saved emotion labels to {out} ({len(emotion_dict)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Download CMU-MOSEI emotion labels")
    parser.add_argument("--output", type=str, default="/content/data/emotion_labels.pkl",
                        help="Output pickle file path")
    parser.add_argument("--download-dir", type=str, default="/content/cmu_mosei_labels",
                        help="Directory to download raw CSD files")
    args = parser.parse_args()

    emotion_dict = download_and_parse(args.download_dir)
    save_labels(emotion_dict, args.output)

    # Print summary stats
    emotions_only = {k: {e: v[e] for e in EMOTION_NAMES[1:]} for k, v in emotion_dict.items()}
    for emo in EMOTION_NAMES[1:]:
        values = [v[emo] for v in emotions_only.values()]
        n_present = sum(1 for v in values if v > 0)
        print(f"  {emo:>10}: {n_present:>6}/{len(values)} ({n_present/len(values)*100:.1f}%) present, "
              f"mean={np.mean(values):.3f}, max={max(values):.1f}")


if __name__ == "__main__":
    main()
