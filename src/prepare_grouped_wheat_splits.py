
"""
AgriAssist-RX
Phase 5 - Step 8: Group-Based Wheat Dataset Splitting

Inputs:
- wheat_train_manifest.csv
- wheat_validation_manifest.csv
- wheat_test_manifest.csv
- full_duplicate_review/all_candidate_pairs.csv

What this script does:
1. Loads the previously cleaned image records.
2. Groups images linked by dHash candidate pairs.
3. Excludes groups with contradictory class labels.
4. Creates reproducible 80/10/10 splits by group.
5. Verifies no flagged candidate pair crosses splits.
6. Saves NEW manifests and an audit report.

CAUTION:
dHash candidates are not always true duplicates.
These are provisional, conservatively grouped splits.
Independent field-level testing is not established.

Original images and manifests are never modified.
"""

import csv
import json
import random

from collections import Counter, defaultdict
from pathlib import Path


# ==========================================
# STEP 1: Configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VISION_DIR = PROJECT_ROOT / "eval" / "vision"

CANDIDATE_FILE = (
    VISION_DIR
    / "full_duplicate_review"
    / "all_candidate_pairs.csv"
)

MANIFEST_FILES = {
    "train": VISION_DIR / "wheat_train_manifest.csv",
    "validation": VISION_DIR / "wheat_validation_manifest.csv",
    "test": VISION_DIR / "wheat_test_manifest.csv",
}

RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10

OUTPUT_PREFIX = "wheat_grouped"

COLUMNS = [
    "path",
    "label",
    "original_split",
    "sha256",
    "group_id",
]


# ==========================================
# STEP 2: Union-Find data structure
# ==========================================

class UnionFind:

    def __init__(self, items):

        self.parent = {
            item: item for item in items
        }

        self.rank = {
            item: 0 for item in items
        }

    def find(self, item):

        if self.parent[item] != item:

            self.parent[item] = self.find(
                self.parent[item]
            )

        return self.parent[item]

    def union(self, first, second):

        root_a = self.find(first)
        root_b = self.find(second)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


# ==========================================
# STEP 3: Load current manifests
# ==========================================

def load_records():

    records = {}

    for split, file_path in MANIFEST_FILES.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing manifest: {file_path}"
            )

        with file_path.open(
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            for row in csv.DictReader(file):

                path = row["path"]

                if path in records:
                    raise ValueError(
                        f"Repeated manifest path: {path}"
                    )

                records[path] = {
                    "path": path,
                    "label": row["label"],
                    "original_split": row.get(
                        "original_split", split
                    ),
                    "sha256": row["sha256"],
                }

    return records


# ==========================================
# STEP 4: Load dHash candidate pairs
# ==========================================

def load_candidate_pairs():

    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"Missing candidate CSV: {CANDIDATE_FILE}"
        )

    with CANDIDATE_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        return list(csv.DictReader(file))


# ==========================================
# STEP 5: Build connected image groups
# ==========================================

def build_groups(records, pairs):

    union_find = UnionFind(records.keys())

    for pair in pairs:

        first = pair["image_1"]
        second = pair["image_2"]

        if first not in records:
            raise ValueError(
                f"Unknown candidate path: {first}"
            )

        if second not in records:
            raise ValueError(
                f"Unknown candidate path: {second}"
            )

        union_find.union(first, second)

    grouped = defaultdict(list)

    for path in sorted(records):

        root = union_find.find(path)

        grouped[root].append(records[path])

    # Assign stable IDs based on sorted member paths.
    groups = []

    ordered = sorted(
        grouped.values(),
        key=lambda group: min(
            item["path"] for item in group
        )
    )

    for index, members in enumerate(ordered, 1):

        group_id = f"G{index:05d}"

        for item in members:
            item["group_id"] = group_id

        groups.append({
            "group_id": group_id,
            "records": members,
            "labels": sorted({
                item["label"] for item in members
            }),
            "size": len(members),
        })

    return groups


# ==========================================
# STEP 6: Separate consistent and
# conflicting-label groups
# ==========================================

def classify_groups(groups):

    clean_groups = []
    excluded_groups = []

    for group in groups:

        if len(group["labels"]) > 1:
            excluded_groups.append(group)
        else:
            clean_groups.append(group)

    return clean_groups, excluded_groups


# ==========================================
# STEP 7: Split complete groups by class
# ==========================================

def create_grouped_splits(clean_groups):

    by_class = defaultdict(list)

    for group in clean_groups:

        label = group["labels"][0]

        by_class[label].append(group)

    rng = random.Random(RANDOM_SEED)

    output = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for label in sorted(by_class):

        groups = sorted(
            by_class[label],
            key=lambda g: g["group_id"]
        )

        rng.shuffle(groups)

        if len(groups) < 3:
            raise ValueError(
                f"Too few independent candidate "
                f"groups for {label}: {len(groups)}"
            )

        total = sum(
            group["size"] for group in groups
        )

        train_target = total * TRAIN_RATIO

        valid_target = total * VALID_RATIO

        counts = {
            "train": 0,
            "validation": 0,
            "test": 0,
        }

        # Keep groups intact.
        # Approximate the intended 80/10/10
        # distribution within each class.
        for group in groups:

            if counts["train"] < train_target:

                split = "train"

            elif counts["validation"] < valid_target:

                split = "validation"

            else:

                split = "test"

            output[split].extend(
                group["records"]
            )

            counts[split] += group["size"]

        # A class must exist in all three splits.
        for split in output:

            if not any(
                item["label"] == label
                for item in output[split]
            ):
                raise ValueError(
                    f"No {label} images in {split}. "
                    "The grouping needs manual review."
                )

        print(
            f"{label:25} "
            f"train={counts['train']:5} "
            f"valid={counts['validation']:4} "
            f"test={counts['test']:4}"
        )

    return output


# ==========================================
# STEP 8: Verify no candidate edge
# crosses the new split boundaries
# ==========================================

def verify_no_cross_split_pairs(output, pairs):

    assignments = {}

    for split, records in output.items():

        for record in records:

            path = record["path"]

            if path in assignments:
                raise RuntimeError(
                    f"Image assigned twice: {path}"
                )

            assignments[path] = split

    violations = []

    for pair in pairs:

        first = pair["image_1"]
        second = pair["image_2"]

        first_split = assignments.get(first)
        second_split = assignments.get(second)

        # Both excluded -> no overlap.
        # One excluded -> no evaluated overlap.
        if first_split is None or second_split is None:
            continue

        if first_split != second_split:

            violations.append(
                pair["pair_id"]
            )

    if violations:
        raise RuntimeError(
            f"Cross-split candidate edges remain: "
            f"{violations[:10]}"
        )

    return len(assignments)


# ==========================================
# STEP 9: Save outputs
# ==========================================

def save_manifests(output):

    for split, records in output.items():

        path = (
            VISION_DIR
            / f"{OUTPUT_PREFIX}_{split}_manifest.csv"
        )

        records = sorted(
            records,
            key=lambda item: (
                item["label"],
                item["group_id"],
                item["path"],
            )
        )

        with path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=COLUMNS
            )

            writer.writeheader()

            writer.writerows(records)

        print(f"Saved: {path.name}")


def save_exclusions(excluded_groups):

    path = (
        VISION_DIR
        / "wheat_grouped_exclusions.json"
    )

    entries = []

    for group in excluded_groups:

        entries.append({
            "group_id": group["group_id"],
            "labels": group["labels"],
            "image_count": group["size"],
            "paths": [
                item["path"]
                for item in group["records"]
            ],
        })

    path.write_text(
        json.dumps(entries, indent=2),
        encoding="utf-8"
    )


# ==========================================
# STEP 10: Main
# ==========================================

def main():

    print("=" * 65)
    print("AGRIASSIST-RX: GROUPED DATASET PREPARATION")
    print("=" * 65)

    records = load_records()
    pairs = load_candidate_pairs()

    print("\nInput images:", len(records))
    print("Flagged candidate pairs:", len(pairs))

    groups = build_groups(records, pairs)

    clean_groups, excluded_groups = (
        classify_groups(groups)
    )

    excluded_images = sum(
        group["size"]
        for group in excluded_groups
    )

    largest_group = max(
        (group["size"] for group in groups),
        default=0
    )

    print("\nGROUP ANALYSIS")
    print("-" * 65)
    print("Total candidate groups:", len(groups))
    print("Consistent-label groups:", len(clean_groups))
    print("Conflicting-label groups:", len(excluded_groups))
    print("Images in excluded groups:", excluded_images)
    print("Largest candidate group:", largest_group)

    print("\nNEW CLASS DISTRIBUTIONS")
    print("-" * 65)

    output = create_grouped_splits(
        clean_groups
    )

    retained = verify_no_cross_split_pairs(
        output,
        pairs
    )

    if retained + excluded_images != len(records):
        raise RuntimeError(
            "Image accounting mismatch."
        )

    print("\nVERIFICATION")
    print("-" * 65)

    print("Flagged cross-split edges remaining: 0")

    save_manifests(output)
    save_exclusions(excluded_groups)

    class_counts = {
        split: dict(sorted(Counter(
            item["label"] for item in items
        ).items()))
        for split, items in output.items()
    }

    summary = {
        "input_images": len(records),
        "candidate_pairs": len(pairs),
        "total_groups": len(groups),
        "consistent_label_groups": len(clean_groups),
        "conflicting_label_groups": len(excluded_groups),
        "excluded_images": excluded_images,
        "retained_images": retained,
        "largest_group_size": largest_group,
        "split_counts": {
            split: len(items)
            for split, items in output.items()
        },
        "class_counts": class_counts,
        "remaining_flagged_cross_split_edges": 0,
        "seed": RANDOM_SEED,
        "limitations": [
            "dHash candidate groups not all manually confirmed",
            "Unreviewed near-duplicates may remain",
            "No independent plant or field IDs",
            "Single-label annotations not fully verified",
            "This is an internal provisional benchmark",
        ],
    }

    summary_path = (
        VISION_DIR
        / "wheat_grouped_split_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\nFINAL SUMMARY")
    print("-" * 65)

    print("Retained images:", retained)
    print("Excluded images:", excluded_images)

    for split, items in output.items():

        print(
            f"{split}: {len(items)} images"
        )

    print("\nSummary:", summary_path)

    print("\nOriginal dataset remains unchanged.")
    print(
        "These are provisional leakage-controlled "
        "manifests, not field-independent validation."
    )


if __name__ == "__main__":
    main()
