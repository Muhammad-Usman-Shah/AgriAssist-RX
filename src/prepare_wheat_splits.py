
"""
AgriAssist-RX
Phase 5: Prepare Clean Wheat Dataset Splits

1. Scan original dataset images.
2. Calculate SHA-256 file hashes.
3. Exclude conflicting-label duplicate groups.
4. Keep one file per unique image hash.
5. Create stratified 80/10/10 splits.
6. Save CSV manifests and a summary report.
7. Verify no exact hash overlaps across splits.

Original images are never modified or deleted.

NOTE:
This removes exact duplicate leakage only.
Near-duplicate and group-leakage checks
are still required before model training.
"""

import csv
import json
import random
import hashlib

from collections import defaultdict, Counter
from pathlib import Path


# ==========================================
# STEP 1: Configuration
# ==========================================

DATASET_ROOT = Path(
    r"D:\AgriAssist-Datasets"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT / "eval" / "vision"
)

SPLITS = {
    "train": DATASET_ROOT / "train",
    "validation": DATASET_ROOT / "valid",
    "test": DATASET_ROOT / "test",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# STEP 2: Normalize class labels
# ==========================================

def normalize_label(name):

    name = name.lower().strip()

    for suffix in [
        "_test_valid",
        "_validation",
        "_training",
        "_valid",
        "_train",
        "_test",
    ]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    return (
        name.replace(" ", "_")
            .replace("-", "_")
    )


# ==========================================
# STEP 3: Calculate image hash
# ==========================================

def calculate_hash(path):

    hasher = hashlib.sha256()

    with path.open("rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            hasher.update(block)

    return hasher.hexdigest()


# ==========================================
# STEP 4: Read the original dataset
# ==========================================

def scan_dataset():

    groups = defaultdict(list)

    total_files = 0

    for split_name, split_path in SPLITS.items():

        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Missing folder: {split_path}"
            )

        print(
            f"\nScanning original {split_name}..."
        )

        for path in sorted(
            split_path.rglob("*")
        ):

            if (
                not path.is_file()
                or path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            label = normalize_label(
                path.parent.name
            )

            file_hash = calculate_hash(path)

            groups[file_hash].append({
                "path": str(path),
                "label": label,
                "original_split": split_name,
                "sha256": file_hash,
            })

            total_files += 1

        print(
            "Total processed so far:",
            total_files
        )

    return groups, total_files


# ==========================================
# STEP 5: Remove ambiguous hash groups
# ==========================================

def select_unique_images(groups):

    clean = []

    excluded = []

    duplicate_copies = 0

    split_priority = {
        "train": 0,
        "validation": 1,
        "test": 2,
    }

    for file_hash, records in groups.items():

        labels = {
            item["label"]
            for item in records
        }

        if len(labels) > 1:

            excluded.append({
                "sha256": file_hash,
                "labels": sorted(labels),
                "files": [
                    item["path"]
                    for item in records
                ],
            })

            continue

        # Prefer a training-folder original
        # only as a deterministic file choice.
        # New split assignment happens later.
        records = sorted(
            records,
            key=lambda item: (
                split_priority[
                    item["original_split"]
                ],
                item["path"],
            )
        )

        representative = records[0]

        clean.append(representative)

        duplicate_copies += len(records) - 1

    return clean, excluded, duplicate_copies


# ==========================================
# STEP 6: Create stratified splits
# ==========================================

def create_splits(clean):

    by_class = defaultdict(list)

    for record in clean:
        by_class[record["label"]].append(
            record
        )

    rng = random.Random(RANDOM_SEED)

    manifests = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for label in sorted(by_class):

        records = sorted(
            by_class[label],
            key=lambda x: x["sha256"]
        )

        rng.shuffle(records)

        count = len(records)

        if count < 10:
            raise ValueError(
                f"Too few images for {label}: {count}"
            )

        n_train = int(
            count * TRAIN_RATIO
        )

        n_valid = int(
            count * VALID_RATIO
        )

        train_records = records[:n_train]

        valid_records = records[
            n_train:n_train + n_valid
        ]

        test_records = records[
            n_train + n_valid:
        ]

        manifests["train"].extend(
            train_records
        )

        manifests["validation"].extend(
            valid_records
        )

        manifests["test"].extend(
            test_records
        )

    return manifests


# ==========================================
# STEP 7: Verify no exact leakage
# ==========================================

def verify_splits(manifests):

    hashes = {
        split: {
            record["sha256"]
            for record in records
        }
        for split, records
        in manifests.items()
    }

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for first, second in pairs:

        overlap = (
            hashes[first] & hashes[second]
        )

        if overlap:
            raise RuntimeError(
                f"Leakage between "
                f"{first} and {second}: "
                f"{len(overlap)} exact hashes"
            )

    print(
        "\nExact hash overlap check: PASSED"
    )


# ==========================================
# STEP 8: Save manifests
# ==========================================

def save_manifests(manifests):

    columns = [
        "path",
        "label",
        "original_split",
        "sha256",
    ]

    for split, records in manifests.items():

        output_path = (
            OUTPUT_DIR
            / f"wheat_{split}_manifest.csv"
        )

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=columns
            )

            writer.writeheader()

            writer.writerows(records)

        print(
            f"Saved {split}: "
            f"{len(records)} images"
        )


# ==========================================
# STEP 9: Main program
# ==========================================

def main():

    print("=" * 60)
    print("AGRIASSIST-RX: CLEAN DATASET PREPARATION")
    print("=" * 60)

    groups, total_files = scan_dataset()

    clean, excluded, redundant = (
        select_unique_images(groups)
    )

    print("\n" + "-" * 60)
    print("DATA CLEANING SUMMARY")
    print("-" * 60)

    print(
        "Original image files:",
        total_files
    )

    print(
        "Unique image hashes:",
        len(groups)
    )

    print(
        "Conflicting hash groups excluded:",
        len(excluded)
    )

    print(
        "Remaining clean unique images:",
        len(clean)
    )

    print(
        "Redundant copies removed among "
        "retained groups:",
        redundant
    )

    manifests = create_splits(clean)

    verify_splits(manifests)

    print("\nNEW DATASET SPLITS")

    save_manifests(manifests)

    class_counts = {}

    for split, records in manifests.items():

        class_counts[split] = dict(
            sorted(
                Counter(
                    record["label"]
                    for record in records
                ).items()
            )
        )

    summary = {
        "original_images": total_files,
        "unique_hashes": len(groups),
        "excluded_conflicting_hash_groups": (
            len(excluded)
        ),
        "retained_unique_images": len(clean),
        "seed": RANDOM_SEED,
        "splits": {
            split: len(records)
            for split, records in manifests.items()
        },
        "class_counts": class_counts,
        "exact_cross_split_overlap": 0,
        "limitations": [
            "Near duplicates not yet checked",
            "No plant or collection-session grouping",
            "Labels not manually verified",
            "Test split is internally held out",
        ],
    }

    summary_path = (
        OUTPUT_DIR / "clean_split_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    conflict_path = (
        OUTPUT_DIR
        / "excluded_label_conflicts.json"
    )

    conflict_path.write_text(
        json.dumps(
            excluded,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("PREPARATION COMPLETE")
    print("=" * 60)

    print("\nSummary saved:")
    print(summary_path)

    print("\nOriginal images remain unchanged.")
    print(
        "Near-duplicate checks are still required "
        "before training."
    )


if __name__ == "__main__":
    main()
