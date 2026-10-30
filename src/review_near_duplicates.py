
"""
AgriAssist-RX
Phase 5: Visual Review of Near-Duplicates

Reads the example pairs from:
eval/vision/near_duplicate_report.json

Creates comparison contact sheets.

Does not alter the original images.

Note: The earlier report stores only 40 examples.
This is an initial visual audit, not a review
of all 603 cross-split candidate pairs.
"""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VISION_DIR = PROJECT_ROOT / "eval" / "vision"

REPORT_PATH = (
    VISION_DIR / "near_duplicate_report.json"
)

OUTPUT_DIR = (
    VISION_DIR / "near_duplicate_review"
)

PAIRS_PER_SHEET = 4

SHEET_WIDTH = 1200
PAIR_HEIGHT = 350

IMAGE_WIDTH = 540
IMAGE_HEIGHT = 250

BACKGROUND = "white"
TEXT_COLOR = "black"


def load_font(size=20):

    paths = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
    ]

    for font_path in paths:

        if Path(font_path).exists():

            return ImageFont.truetype(
                font_path,
                size
            )

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

        canvas.paste(
            image,
            (x, y)
        )

        return canvas


def draw_pair(sheet, pair, position, number, font):

    draw = ImageDraw.Draw(sheet)

    top = position * PAIR_HEIGHT

    draw.text(
        (20, top + 8),
        (
            f"Pair {number} | "
            f"dHash distance: {pair['distance']}"
        ),
        fill=TEXT_COLOR,
        font=font
    )

    paths = [
        pair["image_1"],
        pair["image_2"]
    ]

    labels = [
        pair["label_1"],
        pair["label_2"]
    ]

    splits = [
        pair["split_1"],
        pair["split_2"]
    ]

    for column in range(2):

        x = 20 + column * 590

        caption = (
            f"{splits[column]} | "
            f"{labels[column]}"
        )

        draw.text(
            (x, top + 45),
            caption[:55],
            fill=TEXT_COLOR,
            font=font
        )

        try:

            image = prepare_image(
                paths[column]
            )

            sheet.paste(
                image,
                (x, top + 80)
            )

        except (OSError, ValueError) as error:

            draw.text(
                (x, top + 100),
                f"Image error: {str(error)[:40]}",
                fill="red",
                font=font
            )

    draw.line(
        [
            (20, top + PAIR_HEIGHT - 5),
            (1180, top + PAIR_HEIGHT - 5)
        ],
        fill="gray",
        width=2
    )


def main():

    if not REPORT_PATH.exists():

        print(
            "Missing near-duplicate report:",
            REPORT_PATH
        )
        return

    report = json.loads(
        REPORT_PATH.read_text(
            encoding="utf-8"
        )
    )

    pairs = report.get(
        "example_pairs", []
    )

    if not pairs:

        print("No example pairs available.")
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    font = load_font(19)

    total_sheets = math.ceil(
        len(pairs) / PAIRS_PER_SHEET
    )

    print("=" * 60)
    print("AGRIASSIST-RX: VISUAL DUPLICATE REVIEW")
    print("=" * 60)

    print("Example pairs:", len(pairs))
    print("Comparison sheets:", total_sheets)

    for sheet_index in range(total_sheets):

        start = sheet_index * PAIRS_PER_SHEET

        subset = pairs[
            start:start + PAIRS_PER_SHEET
        ]

        sheet = Image.new(
            "RGB",
            (
                SHEET_WIDTH,
                PAIR_HEIGHT * len(subset)
            ),
            BACKGROUND
        )

        for position, pair in enumerate(subset):

            draw_pair(
                sheet,
                pair,
                position,
                start + position + 1,
                font
            )

        output_path = (
            OUTPUT_DIR
            / f"review_sheet_{sheet_index + 1:02d}.jpg"
        )

        sheet.save(
            output_path,
            quality=90
        )

        print("Created:", output_path.name)

    print("\nAll comparison sheets saved to:")
    print(OUTPUT_DIR)

    print("\nNo original images were modified.")


if __name__ == "__main__":
    main()
