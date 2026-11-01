
"""
AgriAssist-RX
Final MobileNetV3 Test Evaluation

Evaluates the saved model on the grouped test set.

No training is performed.
The existing vision_tool.py model loader
and preprocessing are reused.
"""

import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))

from vision_tool import (
    load_vision_model,
    MODEL_PATH,
    DEVICE,
)


MANIFEST_PATH = (
    ROOT
    / "eval"
    / "vision"
    / "wheat_grouped_test_manifest.csv"
)

OUTPUT_PATH = (
    ROOT
    / "eval"
    / "vision"
    / "mobilenetv3_grouped_test_results.json"
)

BATCH_SIZE = 16


def load_test_records():

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Test manifest not found: {MANIFEST_PATH}"
        )

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        required = {"path", "label"}

        if not required.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                "Manifest must contain path and label."
            )

        records = list(reader)

    return records


@torch.inference_mode()
def evaluate_model():

    print("=" * 60)
    print("AGRIASSIST-RX: FINAL VISION EVALUATION")
    print("=" * 60)

    print("\nLoading saved model...")

    model, preprocess, class_names = (
        load_vision_model()
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    records = load_test_records()

    print("Model:", checkpoint.get("model_name"))
    print("Classes:", len(class_names))
    print("Test records:", len(records))
    print("Device:", DEVICE)

    expected_labels = set(class_names)

    true_labels = []
    predicted_labels = []

    errors = []
    image_batch = []
    label_batch = []

    def process_batch():

        if not image_batch:
            return

        inputs = torch.stack(
            image_batch
        ).to(DEVICE)

        outputs = model(inputs)

        predictions = (
            outputs.argmax(dim=1)
            .cpu()
            .tolist()
        )

        for true_label, predicted_index in zip(
            label_batch,
            predictions,
        ):

            true_labels.append(true_label)

            predicted_labels.append(
                class_names[predicted_index]
            )

        image_batch.clear()
        label_batch.clear()

    for index, record in enumerate(
        records,
        start=1,
    ):

        path = Path(record["path"])
        label = record["label"].strip().lower()

        if label not in expected_labels:

            errors.append({
                "record": index,
                "error": "Unknown class label",
            })

            continue

        if not path.is_file():

            errors.append({
                "record": index,
                "error": "Image file not found",
            })

            continue

        try:

            with Image.open(path) as source:

                image = ImageOps.exif_transpose(
                    source
                ).convert("RGB")

                tensor = preprocess(image)

            image_batch.append(tensor)
            label_batch.append(label)

        except Exception as error:

            errors.append({
                "record": index,
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            })

            continue

        if len(image_batch) >= BATCH_SIZE:
            process_batch()

        if index % 100 == 0:
            print(
                f"Processed {index}/{len(records)}"
            )

    process_batch()

    if not true_labels:
        raise RuntimeError(
            "No test images were evaluated."
        )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=class_names,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=class_names,
    )

    correct = sum(
        actual == predicted
        for actual, predicted in zip(
            true_labels,
            predicted_labels,
        )
    )

    complete = (
        len(errors) == 0
        and len(true_labels) == len(records)
    )

    results = {
        "project": "AgriAssist-RX",
        "evaluation": (
            "MobileNetV3 grouped test evaluation"
        ),
        "model_name": checkpoint.get(
            "model_name"
        ),
        "dataset_type": checkpoint.get(
            "dataset_type"
        ),
        "experiment": checkpoint.get(
            "experiment"
        ),
        "checkpoint_epoch": checkpoint.get(
            "epoch"
        ),
        "number_of_classes": len(class_names),
        "class_names": class_names,
        "total_manifest_records": len(records),
        "evaluated_images": len(true_labels),
        "correct_predictions": correct,
        "error_count": len(errors),
        "complete_evaluation": complete,
        "test_accuracy": round(
            float(accuracy), 6
        ),
        "test_accuracy_percent": round(
            float(accuracy) * 100, 2
        ),
        "test_macro_f1": round(
            float(macro_f1), 6
        ),
        "validation_macro_f1": checkpoint.get(
            "best_val_macro_f1"
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "errors": errors,
        "limitations": [
            "Internal provisional grouped test set",
            "No independent field or plant IDs",
            "Some near-duplicates may remain",
            "Model training provenance is not "
            "fully verified by this evaluation",
            "Not a field-validated diagnostic model",
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print("Total images:", len(records))
    print("Evaluated images:", len(true_labels))
    print("Correct predictions:", correct)

    print(
        "Test accuracy:",
        f"{accuracy * 100:.2f}%"
    )

    print(
        "Test macro-F1:",
        f"{macro_f1:.4f}"
    )

    print("Image errors:", len(errors))

    print(
        "Evaluation complete:",
        complete
    )

    print("\nPer-class results:")

    for name in class_names:

        item = report[name]

        print(
            f"{name:22s} "
            f"F1={item['f1-score']:.3f} "
            f"Support={int(item['support'])}"
        )

    print("\nSaved to:")
    print(OUTPUT_PATH)

    if not complete:
        print(
            "\nWARNING: Some images were not "
            "evaluated. Do not report this as "
            "the full test-set accuracy."
        )


if __name__ == "__main__":
    evaluate_model()
