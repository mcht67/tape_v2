#!/usr/bin/env python3
"""
Fetch the example clips used by plot_polyphony_range.py from the
mcht67/PolyBirdMix POW_soundscape_test config (test_5s split) and store them
under plots/data/polyphony_range/: one row where the polyphony lower bound is
set by unique-species count, one where it's set by true temporal overlap.
Row indices were found by scanning the split for small, clean cases of each
scenario (few call events, no confounding extra overlaps/species).

Usage:
    complete-venv/bin/python source/scripts/fetch_polyphony_range_example.py \\
        [--out-dir plots/data/polyphony_range]
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
import soundfile as sf

POW_SOUNDSCAPE_TEST_FILES = [
    f"POW/POW_soundscape_test/test_5s-0000{i}-of-00003.parquet" for i in range(3)
]

# (dataset row index, output stem) -- row 65: overlap=1 < unique species=2, so
# the species count sets the lower bound; row 1638: overlap=3 > unique
# species=2, so temporal overlap sets it instead.
ROWS = [
    (65, "example", "annotations.json"),
    (1638, "example_overlap", "annotations_overlap.json"),
]


def fetch_row(dataset, species_names, row_idx, out_dir, audio_stem, annotations_name):
    row = dataset[row_idx]
    codes = [species_names[i] for i in row["ebird_code_multilabel"]]
    audio = row["audio"]

    sf.write(out_dir / f"{audio_stem}.wav", audio["array"], audio["sampling_rate"])

    meta = {
        "source": f"mcht67/PolyBirdMix POW_soundscape_test test_5s split, row index {row_idx}",
        "filepath": row["filepath"],
        "segment_start": row["segment_start"],
        "segment_end": row["segment_end"],
        "sampling_rate": audio["sampling_rate"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "low_freq": row["low_freq"],
        "high_freq": row["high_freq"],
        "ebird_code": codes,
        "min_polyphony": row["min_polyphony"],
        "max_polyphony": row["max_polyphony"],
    }
    (out_dir / annotations_name).write_text(json.dumps(meta, indent=2))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/data/polyphony_range"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("mcht67/PolyBirdMix", data_files={"test_5s": POW_SOUNDSCAPE_TEST_FILES})["test_5s"]
    species_names = dataset.features["ebird_code_multilabel"].feature.names

    for row_idx, audio_stem, annotations_name in ROWS:
        meta = fetch_row(dataset, species_names, row_idx, args.out_dir, audio_stem, annotations_name)
        print(f"row {row_idx} -> {audio_stem}.wav / {annotations_name}: {meta['ebird_code']}, "
              f"min/max polyphony = {meta['min_polyphony']}/{meta['max_polyphony']}")


if __name__ == "__main__":
    main()
