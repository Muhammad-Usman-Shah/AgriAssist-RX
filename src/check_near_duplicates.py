
"""
AgriAssist-RX
Phase 5: Near-Duplicate Image Inspection

Uses 64-bit difference hashing (dHash).

Reads cleaned train/validation/test manifests.

Detects visually similar image fingerprints,
especially across different dataset splits.

Does not modify, copy, or delete any images.

IMPORTANT:
A dHash match is a candidate for review,
not proof that two photographs are identical.
"""

import csv
import json

from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


# ==========================================
# STEP 1: Configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VISION_DIR = PROJECT_ROOT / "eval" / "vision"

MANIFESTS = {
    "train": VISION_DIR / "wheat_train_manifest.csv",
    "validation": VISION_DIR / "wheat_validation_manifest.csv",
    "test": VISION_DIR / "wheat_test_manifest.csv",
}

REPORT_PATH = (
    VISION_DIR / "near_duplicate_report.json"
)

# Similarity threshold.
# Smaller value = stricter matching.
MAX_DISTANCE = 3

# Maximum number of example pairs in report.
MAX_EXAMPLES = 40


# ==========================================
# STEP 2: Read the cleaned manifests
# ==========================================

def load_records():

    records = []

    for split, manifest_path in MANIFESTS.items():

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest missing: {manifest_path}"
            )

        with manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                records.append({
                    "path": row["path"],
                    "label": row["label"],
                    "split": split,
                    "sha256": row["sha256"],
                })

    return records


# ==========================================
# STEP 3: Calculate perceptual difference hash
# ==========================================

def calculate_dhash(path):

    with Image.open(path) as original:

        # Respect image rotation information.
        image = ImageOps.exif_transpose(original)

        # Convert to grayscale.
        image = image.convert("L")

        # Create 9 columns x 8 rows.
        image = image.resize(
            (9, 8),
            Image.Resampling.LANCZOS
        )

        pixels = list(image.getdata())

    result = 0

    # Compare horizontal neighboring pixels.
    for y in range(8):

        for x in range(8):

            left = pixels[y * 9 + x]
            right = pixels[y * 9 + x + 1]

            result = (result << 1) | int(
                left > right
            )

    return result


# ==========================================
# STEP 4: Calculate fingerprint distance
# ==========================================

def hamming_distance(first, second):

    return (first ^ second).bit_count()


# ==========================================
# STEP 5: Find candidate fingerprints
# ==========================================

def get_segments(image_hash):

    # Split the 64-bit hash into four
    # separate 16-bit blocks.
    return [
        (image_hash >> shift) & 0xFFFF
        for shift in (0, 16, 32, 48)
    ]


def find_candidates(image_hash, index):

    candidates = set()

    for position, segment in enumerate(
        get_segments(image_hash)
    ):

        key = (position, segment)

        candidates.update(
            index.get(key, set())
        )

    return candidates


def add_to_index(image_hash, record_id, index):

    for position, segment in enumerate(
        get_segments(image_hash)
    ):

        key = (position, segment)

        index[key].add(record_id)


# ==========================================
# STEP 6: Inspect all cleaned images
# ==========================================

def main():

    print("=" * 60)
    print("AGRIASSIST-RX: NEAR-DUPLICATE INSPECTION")
    print("=" * 60)

    records = load_records()

    print("\nImages to inspect:", len(records))

    hashes = []
    index = defaultdict(set)

    cross_split_pairs = 0
    conflicting_label_pairs = 0

    affected_images = set()

    split_combinations = Counter()

    examples = []
    errors = []

    for i, record in enumerate(records):

        path = Path(record["path"])

        try:
            image_hash = calculate_dhash(path)

        except (OSError, ValueError) as error:

            errors.append({
                "path": str(path),
                "error": str(error),
            })

            hashes.append(None)
            continue

        # Retrieve possible matches without
        # comparing every image to every other.
        candidates = find_candidates(
            image_hash,
            index
        )

        for candidate_id in candidates:

            previous = records[candidate_id]
            previous_hash = hashes[candidate_id]

            distance = hamming_distance(
                image_hash,
                previous_hash
            )

            if distance > MAX_DISTANCE:
                continue

            different_splits = (
                record["split"]
                != previous["split"]
            )

            different_labels = (
                record["label"]
                != previous["label"]
            )

            # Record cross-split and
            # conflicting-label candidates.
            if not (
                different_splits
                or different_labels
            ):
                continue

            if different_splits:

                cross_split_pairs += 1

                affected_images.add(i)
                affected_images.add(candidate_id)

                combination = " + ".join(
                    sorted([
                        record["split"],
                        previous["split"],
                    ])
                )

                split_combinations[
                    combination
                ] += 1

            if different_labels:
                conflicting_label_pairs += 1

            if len(examples) < MAX_EXAMPLES:

                examples.append({
                    "image_1": previous["path"],
                    "image_2": record["path"],
                    "split_1": previous["split"],
                    "split_2": record["split"],
                    "label_1": previous["label"],
                    "label_2": record["label"],
                    "distance": distance,
                    "cross_split": different_splits,
                    "different_labels": different_labels,
                })

        # Save this image for future comparisons.
        hashes.append(image_hash)

        add_to_index(
            image_hash,
            i,
            index
        )

        if (i + 1) % 500 == 0:

            print(
                f"Checked {i + 1}/{len(records)}",
                flush=True
            )

    # ======================================
    # STEP 7: Save the report
    # ======================================

    report = {
        "total_manifest_images": len(records),
        "hash_method": "64-bit dHash",
        "maximum_hamming_distance": MAX_DISTANCE,
        "cross_split_candidate_pairs": (
            cross_split_pairs
        ),
        "different_label_candidate_pairs": (
            conflicting_label_pairs
        ),
        "images_in_cross_split_pairs": len(
            affected_images
        ),
        "split_combinations": dict(
            split_combinations
        ),
        "unreadable_images": errors,
        "example_pairs": examples,
        "notes": [
            "Similarity candidates require review.",
            "No source images were modified.",
            "dHash may miss cropped or rotated near-duplicates.",
            "No plant or acquisition-group independence established.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("NEAR-DUPLICATE INSPECTION RESULTS")
    print("=" * 60)

    print(
        "Images inspected:",
        len(records)
    )

    print(
        "Cross-split candidate pairs:",
        cross_split_pairs
    )

    print(
        "Different-label candidate pairs:",
        conflicting_label_pairs
    )

    print(
        "Images involved in cross-split pairs:",
        len(affected_images)
    )

    print(
        "Images with reading errors:",
        len(errors)
    )

    print("\nCross-split combinations:")

    for combination, count in sorted(
        split_combinations.items()
    ):

        print(
            f"  {combination}: {count}"
        )

    print("\nReport saved to:")
    print(REPORT_PATH)

    print("\nOriginal images remain unchanged.")


if __name__ == "__main__":
    main()
