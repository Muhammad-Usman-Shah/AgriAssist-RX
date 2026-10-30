
"""
AgriAssist-RX
Phase 5: Image Quality and Exact Duplicate Check

Checks:
1. Total image files
2. Corrupted / unreadable images
3. Exact duplicate images using SHA-256
4. Duplicates across train, valid, and test
5. Conflicting labels on identical images

Does NOT modify or delete images.
"""

import json
import hashlib

from pathlib import Path
from collections import defaultdict

from PIL import Image, UnidentifiedImageError


# ==========================================
# STEP 1: Dataset paths
# ==========================================

DATASET_ROOT = Path(
    r"D:\AgriAssist-Datasets"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_PATH = (
    PROJECT_ROOT
    / "eval"
    / "vision"
    / "image_quality_report.json"
)

SPLITS = {
    "train": DATASET_ROOT / "train",
    "validation": DATASET_ROOT / "valid",
    "test": DATASET_ROOT / "test",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ==========================================
# STEP 2: Normalize labels
# ==========================================

def normalize_label(folder_name):

    name = folder_name.lower().strip()

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

    return name.replace(" ", "_").replace("-", "_")


# ==========================================
# STEP 3: Calculate a file hash
# ==========================================

def calculate_hash(path):

    sha256 = hashlib.sha256()

    with path.open("rb") as file:

        while True:

            block = file.read(1024 * 1024)

            if not block:
                break

            sha256.update(block)

    return sha256.hexdigest()


# ==========================================
# STEP 4: Verify that image can be opened
# ==========================================

def verify_image(path):

    try:

        with Image.open(path) as image:

            image.verify()

        return True, None

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
    ) as error:

        return False, str(error)


# ==========================================
# STEP 5: Inspect the dataset
# ==========================================

def main():

    print("=" * 60)
    print("AGRIASSIST-RX: IMAGE QUALITY CHECK")
    print("=" * 60)

    records = []
    corrupted = []

    for split_name, split_path in SPLITS.items():

        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Missing split: {split_path}"
            )

        image_paths = sorted(
            path
            for path in split_path.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )

        print(
            f"\nChecking {split_name}: "
            f"{len(image_paths)} images",
            flush=True
        )

        for index, path in enumerate(
            image_paths, start=1
        ):

            label = normalize_label(
                path.parent.name
            )

            is_valid, error = verify_image(path)

            if not is_valid:

                corrupted.append({
                    "path": str(path),
                    "split": split_name,
                    "error": error,
                })

                continue

            file_hash = calculate_hash(path)

            records.append({
                "path": str(path),
                "split": split_name,
                "label": label,
                "sha256": file_hash,
            })

            if index % 500 == 0:

                print(
                    f"  Checked {index}/"
                    f"{len(image_paths)}",
                    flush=True
                )

    # ======================================
    # STEP 6: Group identical files
    # ======================================

    hash_groups = defaultdict(list)

    for record in records:

        hash_groups[
            record["sha256"]
        ].append(record)

    duplicate_groups = []

    for file_hash, group in hash_groups.items():

        if len(group) < 2:
            continue

        splits = sorted({
            item["split"]
            for item in group
        })

        labels = sorted({
            item["label"]
            for item in group
        })

        duplicate_groups.append({
            "sha256": file_hash,
            "count": len(group),
            "splits": splits,
            "labels": labels,
            "cross_split": len(splits) > 1,
            "conflicting_labels": len(labels) > 1,
            "files": [
                item["path"]
                for item in group
            ],
        })

    cross_split = [
        group for group in duplicate_groups
        if group["cross_split"]
    ]

    conflicting_labels = [
        group for group in duplicate_groups
        if group["conflicting_labels"]
    ]

    # ======================================
    # STEP 7: Save the report
    # ======================================

    report = {
        "total_valid_images": len(records),
        "corrupted_images": len(corrupted),
        "duplicate_groups": len(duplicate_groups),
        "cross_split_duplicate_groups": len(cross_split),
        "conflicting_label_groups": len(
            conflicting_labels
        ),
        "corrupted_files": corrupted,
        "duplicates": duplicate_groups,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("IMAGE QUALITY SUMMARY")
    print("=" * 60)

    print(
        "Valid image files:",
        len(records)
    )

    print(
        "Corrupted images:",
        len(corrupted)
    )

    print(
        "Exact duplicate groups:",
        len(duplicate_groups)
    )

    print(
        "Cross-split duplicate groups:",
        len(cross_split)
    )

    print(
        "Conflicting-label duplicate groups:",
        len(conflicting_labels)
    )

    if cross_split:

        print("\nExample cross-split duplicate:")

        for path in cross_split[0]["files"][:5]:
            print(" ", path)

    if corrupted:

        print("\nExample corrupted image:")
        print(corrupted[0]["path"])

    print("\nFull report saved to:")
    print(REPORT_PATH)

    print("\nNo images were modified or deleted.")


if __name__ == "__main__":
    main()
