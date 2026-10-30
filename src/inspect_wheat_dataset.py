
"""
AgriAssist-RX
Phase 5: Wheat Dataset Inspection

What this program does:
1. Finds train, validation and test folders.
2. Counts images in each class.
3. Displays class distributions.
4. Identifies inconsistent class folder names.
5. Saves a JSON inventory report.

No training happens in this step.
"""

import json
from pathlib import Path
from collections import Counter


# ==========================================
# STEP 1: Set dataset location
# ==========================================

DATASET_ROOT = Path(
    r"D:\AgriAssist-Datasets"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_PATH = (
    PROJECT_ROOT
    / "eval"
    / "vision"
    / "dataset_inventory.json"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

SPLIT_NAMES = {
    "train": "train",
    "training": "train",
    "valid": "validation",
    "validation": "validation",
    "val": "validation",
    "test": "test",
    "testing": "test",
}


# ==========================================
# STEP 2: Standardize class names
# ==========================================

def normalize_class(name):

    name = name.lower().strip()

    # Some folders include split suffixes.
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

    name = name.replace(" ", "_")
    name = name.replace("-", "_")

    return name


# ==========================================
# STEP 3: Find dataset split folders
# ==========================================

def find_splits(root):

    found = {}

    for folder in root.rglob("*"):

        if not folder.is_dir():
            continue

        split = SPLIT_NAMES.get(
            folder.name.lower()
        )

        if split is not None:
            found.setdefault(split, []).append(folder)

    return found


# ==========================================
# STEP 4: Count images by class
# ==========================================

def inspect_split(folder):

    counts = Counter()
    original_names = {}

    for class_folder in sorted(folder.iterdir()):

        if not class_folder.is_dir():
            continue

        label = normalize_class(
            class_folder.name
        )

        count = sum(
            1
            for path in class_folder.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )

        counts[label] += count

        original_names.setdefault(
            label, []
        ).append(class_folder.name)

    return {
        "folder": str(folder),
        "total_images": sum(counts.values()),
        "class_counts": dict(counts),
        "original_folder_names": original_names,
    }


# ==========================================
# STEP 5: Main inspection program
# ==========================================

def main():

    print("=" * 60)
    print("AGRIASSIST-RX: WHEAT DATASET INSPECTION")
    print("=" * 60)

    if not DATASET_ROOT.exists():
        print(
            "Dataset path not found:",
            DATASET_ROOT
        )
        return

    splits = find_splits(DATASET_ROOT)

    if not splits:
        print(
            "\nNo standard train/valid/test "
            "folders were found."
        )

        print("\nTop-level folders:")

        for folder in DATASET_ROOT.iterdir():
            if folder.is_dir():
                print(" ", folder)

        print(
            "\nWe need to adjust the dataset "
            "path or split folder names."
        )
        return

    report = {
        "dataset_root": str(DATASET_ROOT),
        "splits": {},
    }

    for split_name in [
        "train", "validation", "test"
    ]:

        folders = splits.get(
            split_name, []
        )

        if not folders:
            print(
                f"\nMissing split: {split_name}"
            )
            continue

        for index, folder in enumerate(folders):

            result = inspect_split(folder)

            report_key = split_name

            if len(folders) > 1:
                report_key = (
                    f"{split_name}_{index + 1}"
                )

            report["splits"][report_key] = result

            print("\n" + "-" * 60)
            print("SPLIT:", report_key.upper())
            print("Folder:", folder)
            print(
                "Total images:",
                result["total_images"]
            )

            print("\nClass distribution:")

            for label, count in sorted(
                result["class_counts"].items()
            ):
                print(
                    f"  {label:25} {count:6}"
                )

    print("\n" + "=" * 60)

    print("CLASS CONSISTENCY CHECK")

    class_sets = {
        name: set(
            result["class_counts"].keys()
        )
        for name, result
        in report["splits"].items()
    }

    all_classes = set().union(
        *class_sets.values()
    ) if class_sets else set()

    print(
        "Unique normalized classes:",
        len(all_classes)
    )

    for split_name, labels in class_sets.items():

        missing = sorted(
            all_classes - labels
        )

        if missing:
            print(
                f"{split_name} missing:",
                missing
            )
        else:
            print(
                f"{split_name}: all classes present"
            )

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

    print("\nReport saved to:")
    print(REPORT_PATH)

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
