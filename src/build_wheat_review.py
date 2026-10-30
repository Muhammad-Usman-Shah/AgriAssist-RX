
"""
AgriAssist-RX
Phase 5 - Step 7: Full Near-Duplicate Review

Reads the cleaned dataset manifests.
Recomputes 64-bit dHash fingerprints.

Outputs:
1. All cross-split candidate pairs (CSV)
2. All different-label candidate pairs (CSV)
3. Side-by-side image review sheets

The review sheets sample up to 10 pairs
at each dHash distance (0, 1, 2, 3).

No original images are changed.
No dataset split is changed.

IMPORTANT:
A perceptual-hash match requires
visual verification.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


# ==========================================
# 1. PROJECT SETTINGS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VISION_DIR = PROJECT_ROOT / "eval" / "vision"

OUTPUT_DIR = VISION_DIR / "full_duplicate_review"

MANIFESTS = {
    "train": VISION_DIR / "wheat_train_manifest.csv",
    "validation": VISION_DIR / "wheat_validation_manifest.csv",
    "test": VISION_DIR / "wheat_test_manifest.csv",
}

MAX_DISTANCE = 3

PAIRS_PER_SHEET = 4
SAMPLES_PER_DISTANCE = 10

IMAGE_WIDTH = 520
IMAGE_HEIGHT = 245

SHEET_WIDTH = 1160
PAIR_HEIGHT = 340


# ==========================================
# 2. LOAD DATASET MANIFESTS
# ==========================================

def load_records():

    records = []

    for split, file_path in MANIFESTS.items():

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing manifest: {file_path}"
            )

        with file_path.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                path = Path(row["path"])

                if not path.is_file():
                    raise FileNotFoundError(
                        f"Image not found: {path}"
                    )

                records.append({
                    "path": str(path),
                    "label": row["label"],
                    "split": split,
                })

    return records


# ==========================================
# 3. COMPUTE PERCEPTUAL HASH
# ==========================================

def calculate_dhash(path):

    with Image.open(path) as original:

        image = ImageOps.exif_transpose(
            original
        ).convert("L")

        image = image.resize(
            (9, 8),
            Image.Resampling.LANCZOS
        )

        pixels = list(image.getdata())

    value = 0

    for y in range(8):
        for x in range(8):

            left = pixels[y * 9 + x]
            right = pixels[y * 9 + x + 1]

            value = (
                (value << 1)
                | int(left > right)
            )

    return value


def distance(hash1, hash2):
    return (hash1 ^ hash2).bit_count()


def get_hash_segments(value):

    return [
        (value >> shift) & 0xFFFF
        for shift in (0, 16, 32, 48)
    ]


# ==========================================
# 4. FIND ALL FLAGGED PAIRS
# ==========================================

def find_pairs(records):

    index = defaultdict(set)
    hashes = []

    pairs = []

    print("\nCalculating image fingerprints...")

    for i, record in enumerate(records):

        image_hash = calculate_dhash(
            record["path"]
        )

        candidates = set()

        segments = get_hash_segments(
            image_hash
        )

        for position, segment in enumerate(segments):

            candidates.update(
                index[(position, segment)]
            )

        for previous_id in candidates:

            previous = records[previous_id]

            dist = distance(
                image_hash,
                hashes[previous_id]
            )

            if dist > MAX_DISTANCE:
                continue

            cross_split = (
                record["split"] != previous["split"]
            )

            label_conflict = (
                record["label"] != previous["label"]
            )

            if not (cross_split or label_conflict):
                continue

            pair = {
                "pair_id": "",
                "distance": dist,
                "split_1": previous["split"],
                "label_1": previous["label"],
                "image_1": previous["path"],
                "split_2": record["split"],
                "label_2": record["label"],
                "image_2": record["path"],
                "cross_split": cross_split,
                "different_labels": label_conflict,
                "decision": "",
            }

            pairs.append(pair)

        hashes.append(image_hash)

        for position, segment in enumerate(segments):

            index[(position, segment)].add(i)

        if (i + 1) % 500 == 0:

            print(
                f"Processed {i + 1}/{len(records)}",
                flush=True
            )

    # Sort pairs in a reproducible order.
    pairs.sort(
        key=lambda p: (
            p["distance"],
            p["split_1"],
            p["split_2"],
            p["image_1"],
            p["image_2"],
        )
    )

    for i, pair in enumerate(pairs, start=1):
        pair["pair_id"] = f"P{i:05d}"

    return pairs


# ==========================================
# 5. SAVE COMPLETE CSV REPORTS
# ==========================================

COLUMNS = [
    "pair_id",
    "distance",
    "split_1",
    "label_1",
    "image_1",
    "split_2",
    "label_2",
    "image_2",
    "cross_split",
    "different_labels",
    "decision",
]


def save_csv(path, pairs):

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=COLUMNS
        )

        writer.writeheader()
        writer.writerows(pairs)


# ==========================================
# 6. SELECT REVIEW EXAMPLES
# ==========================================

def select_examples(pairs):

    selected = []

    # Review different similarity levels,
    # not just the first 40 pairs.
    for dist in range(MAX_DISTANCE + 1):

        matching = [
            p for p in pairs
            if p["distance"] == dist
        ]

        selected.extend(
            matching[:SAMPLES_PER_DISTANCE]
        )

    return selected


# ==========================================
# 7. DRAW SIDE-BY-SIDE SHEETS
# ==========================================

def load_font(size=19):

    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ]

    for path in candidates:

        if Path(path).is_file():
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()


def prepare_image(path):

    with Image.open(path) as source:

        image = ImageOps.exif_transpose(
            source
        ).convert("RGB")

        image.thumbnail(
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            Image.Resampling.LANCZOS
        )

        canvas = Image.new(
            "RGB",
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            "white"
        )

        x = (IMAGE_WIDTH - image.width) // 2
        y = (IMAGE_HEIGHT - image.height) // 2

        canvas.paste(image, (x, y))

        return canvas


def create_sheets(name, pairs):

    folder = OUTPUT_DIR / name
    folder.mkdir(parents=True, exist_ok=True)

    font = load_font(19)
    small_font = load_font(15)

    number_of_sheets = math.ceil(
        len(pairs) / PAIRS_PER_SHEET
    )

    for sheet_index in range(number_of_sheets):

        batch = pairs[
            sheet_index * PAIRS_PER_SHEET:
            (sheet_index + 1) * PAIRS_PER_SHEET
        ]

        sheet = Image.new(
            "RGB",
            (
                SHEET_WIDTH,
                PAIR_HEIGHT * len(batch)
            ),
            "white"
        )

        draw = ImageDraw.Draw(sheet)

        for row, pair in enumerate(batch):

            top = row * PAIR_HEIGHT

            heading = (
                f"{pair['pair_id']} | "
                f"dHash distance: {pair['distance']}"
            )

            draw.text(
                (20, top + 8),
                heading,
                font=font,
                fill="black"
            )

            for column in (1, 2):

                x = 20 + (column - 1) * 580

                label = (
                    f"{pair[f'split_{column}']} | "
                    f"{pair[f'label_{column}']}"
                )

                draw.text(
                    (x, top + 42),
                    label,
                    font=font,
                    fill="black"
                )

                path = pair[f"image_{column}"]

                image = prepare_image(path)

                sheet.paste(
                    image,
                    (x, top + 70)
                )

                draw.text(
                    (x, top + 320),
                    Path(path).name[:60],
                    font=small_font,
                    fill="black"
                )

            draw.line(
                [
                    (20, top + PAIR_HEIGHT - 2),
                    (1140, top + PAIR_HEIGHT - 2)
                ],
                fill="gray",
                width=1
            )

        output_path = (
            folder / f"sheet_{sheet_index + 1:02d}.jpg"
        )

        sheet.save(output_path, quality=90)

    print(
        f"Created {number_of_sheets} sheets "
        f"in {name}"
    )


# ==========================================
# 8. MAIN PROGRAM
# ==========================================

def main():

    print("=" * 60)
    print("AGRIASSIST-RX: FULL DUPLICATE REVIEW")
    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    records = load_records()

    print("Images:", len(records))

    pairs = find_pairs(records)

    cross_split = [
        p for p in pairs
        if p["cross_split"]
    ]

    conflicting = [
        p for p in pairs
        if p["different_labels"]
    ]

    train_test = [
        p for p in cross_split
        if {
            p["split_1"], p["split_2"]
        } == {"train", "test"}
    ]

    train_valid = [
        p for p in cross_split
        if {
            p["split_1"], p["split_2"]
        } == {"train", "validation"}
    ]

    valid_test = [
        p for p in cross_split
        if {
            p["split_1"], p["split_2"]
        } == {"validation", "test"}
    ]

    save_csv(
        OUTPUT_DIR / "all_candidate_pairs.csv",
        pairs
    )

    save_csv(
        OUTPUT_DIR / "cross_split_candidates.csv",
        cross_split
    )

    save_csv(
        OUTPUT_DIR / "label_conflict_candidates.csv",
        conflicting
    )

    review_categories = {
        "01_train_test": train_test,
        "02_train_validation": train_valid,
        "03_validation_test": valid_test,
        "04_label_conflicts": conflicting,
    }

    for name, category_pairs in review_categories.items():

        examples = select_examples(
            category_pairs
        )

        create_sheets(
            name,
            examples
        )

    print("\n" + "=" * 60)
    print("FULL REVIEW SUMMARY")
    print("=" * 60)

    print("All flagged pairs:", len(pairs))
    print("Cross-split pairs:", len(cross_split))
    print("Different-label pairs:", len(conflicting))
    print("Train-test:", len(train_test))
    print("Train-validation:", len(train_valid))
    print("Validation-test:", len(valid_test))

    print("\nOutputs saved to:")
    print(OUTPUT_DIR)

    print("\nOriginal dataset remains unchanged.")
    print(
        "No cleaning decisions have been "
        "made automatically."
    )


if __name__ == "__main__":
    main()
