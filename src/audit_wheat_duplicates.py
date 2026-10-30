
"""
AgriAssist-RX
Phase 5: Duplicate and Label Conflict Audit

Reads the existing image-quality report.

Reports:
1. Exact duplicate groups
2. Number of redundant image copies
3. Cross-split duplicate groups
4. Conflicting-label groups
5. Example cases for manual review

No images are deleted or modified.
"""

import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "eval"
    / "vision"
    / "image_quality_report.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "eval"
    / "vision"
    / "duplicate_audit_summary.json"
)


def main():

    print("=" * 60)
    print("AGRIASSIST-RX: DUPLICATE DATASET AUDIT")
    print("=" * 60)

    if not INPUT_PATH.exists():
        print("Image quality report not found.")
        return

    report = json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )

    groups = report.get("duplicates", [])

    total_duplicate_files = sum(
        group["count"]
        for group in groups
    )

    redundant_copies = sum(
        group["count"] - 1
        for group in groups
    )

    cross_split = [
        group for group in groups
        if group["cross_split"]
    ]

    conflicting = [
        group for group in groups
        if group["conflicting_labels"]
    ]

    overlapping = [
        group for group in groups
        if (
            group["cross_split"]
            and group["conflicting_labels"]
        )
    ]

    print("\nDATASET SUMMARY")
    print("-" * 60)

    print(
        "Valid image files:",
        report["total_valid_images"]
    )

    print(
        "Exact duplicate groups:",
        len(groups)
    )

    print(
        "Files belonging to duplicate groups:",
        total_duplicate_files
    )

    print(
        "Redundant file copies:",
        redundant_copies
    )

    print(
        "Cross-split duplicate groups:",
        len(cross_split)
    )

    print(
        "Conflicting-label groups:",
        len(conflicting)
    )

    print(
        "Groups with both problems:",
        len(overlapping)
    )

    # Count combinations of affected splits.
    split_counter = Counter()

    for group in cross_split:

        combination = " + ".join(
            sorted(group["splits"])
        )

        split_counter[combination] += 1

    print("\nCROSS-SPLIT COMBINATIONS")
    print("-" * 60)

    for combination, count in sorted(
        split_counter.items()
    ):
        print(f"{combination}: {count}")

    # Count conflicting label combinations.
    label_counter = Counter()

    for group in conflicting:

        combination = " | ".join(
            sorted(group["labels"])
        )

        label_counter[combination] += 1

    print("\nMOST COMMON LABEL CONFLICTS")
    print("-" * 60)

    for labels, count in label_counter.most_common(10):
        print(f"{labels}: {count}")

    print("\nEXAMPLE CONFLICTING IMAGES")
    print("-" * 60)

    for index, group in enumerate(
        conflicting[:3],
        start=1
    ):

        print(f"\nExample {index}")

        print(
            "Labels:",
            ", ".join(group["labels"])
        )

        print(
            "Splits:",
            ", ".join(group["splits"])
        )

        for filename in group["files"][:5]:
            print(" ", filename)

    summary = {
        "valid_images": report["total_valid_images"],
        "duplicate_groups": len(groups),
        "files_in_duplicate_groups": total_duplicate_files,
        "redundant_file_copies": redundant_copies,
        "cross_split_duplicate_groups": len(cross_split),
        "conflicting_label_groups": len(conflicting),
        "both_cross_split_and_conflicting": len(overlapping),
        "cross_split_combinations": dict(split_counter),
        "most_common_label_conflicts": [
            {
                "labels": labels,
                "groups": count
            }
            for labels, count
            in label_counter.most_common(20)
        ],
        "example_conflicts": conflicting[:5],
    }

    OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\nReport saved to:")
    print(OUTPUT_PATH)

    print("\nNo images were modified.")


if __name__ == "__main__":
    main()
