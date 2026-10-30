
"""
AgriAssist-RX - Phase 5
Local Wheat Image Classification Tool

Loads a fine-tuned MobileNetV3-Small checkpoint.
Predicts the top three wheat conditions.

Usage:
    python src/vision_tool.py "D:\\path\\to\\photo.jpg"

This is an experimental classifier.
Predictions are not confirmed diagnoses.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_PATH = Path(
    r"D:\AgriAssist-Datasets\models"
    r"\mobilenetv3_finetuned_best.pth"
)

DEVICE = torch.device("cpu")

ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}

MAX_IMAGE_BYTES = 25 * 1024 * 1024

# Demonstration screening values only.
# These are NOT calibrated reliability thresholds.
MIN_TOP_SCORE = 0.70
MIN_SCORE_MARGIN = 0.15


# --------------------------------------------------
# 2. Load model once
# --------------------------------------------------

def load_vision_model(model_path=MODEL_PATH):

    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    class_names = list(checkpoint["class_names"])
    num_classes = int(checkpoint["num_classes"])

    if len(class_names) != num_classes:
        raise ValueError(
            "Checkpoint class names do not match "
            "the number of classes."
        )

    network = mobilenet_v3_small(weights=None)

    in_features = network.classifier[-1].in_features

    network.classifier[-1] = nn.Linear(
        in_features,
        num_classes,
    )

    network.load_state_dict(
        checkpoint["model_state_dict"]
    )

    network.to(DEVICE)
    network.eval()

    preprocess = (
        MobileNet_V3_Small_Weights.DEFAULT.transforms()
    )

    return network, preprocess, class_names


# --------------------------------------------------
# 3. Single-image prediction
# --------------------------------------------------

@torch.inference_mode()
def predict_wheat_image(
    image_path,
    model=None,
    preprocess=None,
    class_names=None,
):

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported format. Use JPG, PNG, "
            "WEBP, or BMP."
        )

    if image_path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError(
            "Image is larger than the 25 MB limit."
        )

    if (
        model is None
        or preprocess is None
        or class_names is None
    ):
        model, preprocess, class_names = (
            load_vision_model()
        )

    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(
                source
            ).convert("RGB")

    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            f"Cannot open image: {error}"
        ) from error

    tensor = (
        preprocess(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    outputs = model(tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1,
    )[0]

    scores, indices = torch.topk(
        probabilities,
        k=min(3, len(class_names)),
    )

    predictions = []

    for score, index in zip(
        scores.tolist(),
        indices.tolist(),
    ):
        predictions.append({
            "condition": class_names[index],
            "model_score": round(float(score), 6),
        })

    highest = predictions[0]["model_score"]
    second = predictions[1]["model_score"]

    review_recommended = (
        highest < MIN_TOP_SCORE
        or highest - second < MIN_SCORE_MARGIN
    )

    return {
        "status": "prediction_generated",
        "image": str(image_path),
        "top_prediction": predictions[0]["condition"],
        "predictions": predictions,
        "review_recommended": review_recommended,
        "warning": (
            "Experimental wheat-condition classifier. "
            "The model does not verify that the photograph "
            "contains wheat or clearly visible symptoms. "
            "Softmax scores are not calibrated diagnostic "
            "probabilities. A high score does not guarantee "
            "a correct diagnosis. Independent review may "
            "be necessary."
        ),
    }


# --------------------------------------------------
# 4. Command-line interface
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Predict wheat conditions from an image."
        )
    )

    parser.add_argument(
        "image",
        help="Path to a wheat photograph",
    )

    args = parser.parse_args()

    print("Loading MobileNetV3 on CPU...")

    model, preprocess, class_names = (
        load_vision_model()
    )

    result = predict_wheat_image(
        args.image,
        model=model,
        preprocess=preprocess,
        class_names=class_names,
    )

    print("\n" + "=" * 55)
    print("AGRIASSIST-RX: IMAGE PREDICTION")
    print("=" * 55)

    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
