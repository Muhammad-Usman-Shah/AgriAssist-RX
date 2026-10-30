
"""
AgriAssist-RX
Phase 5 - Step 21

LangGraph Computer Vision Integration

This script connects our existing vision_tool.py
to a LangGraph workflow.

Input:
    Local image path

Output:
    Top 3 predicted wheat conditions
    Model scores
    Uncertainty warning

The existing RAG, weather, calculator and
CSV scripts are not modified.

Run from the project root:

    .\\.venv\\Scripts\\python.exe src\\agent_vision.py --image "D:\\path\\to\\image.png"
"""

import argparse
import json

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from vision_tool import (
    load_vision_model,
    predict_wheat_image,
)


# =====================================================
# 1. DEFINE THE AGENT STATE
# =====================================================

class VisionAgentState(TypedDict, total=False):

    image_path: str
    question: str
    route: str
    prediction: dict
    response: str
    error: str


# =====================================================
# 2. LOAD MODEL ONCE
# =====================================================

@lru_cache(maxsize=1)
def get_vision_resources():

    print("Loading wheat classification model...")

    model, preprocess, class_names = (
        load_vision_model()
    )

    return model, preprocess, class_names


# =====================================================
# 3. IMAGE-ANALYSIS NODE
# =====================================================

def vision_node(state: VisionAgentState):

    image_path = state.get(
        "image_path", ""
    ).strip()

    if not image_path:

        return {
            "route": "vision",
            "error": "No image path was provided.",
            "response": (
                "Please provide a wheat image path."
            ),
        }

    try:

        model, preprocess, class_names = (
            get_vision_resources()
        )

        result = predict_wheat_image(
            image_path,
            model=model,
            preprocess=preprocess,
            class_names=class_names,
        )

        predictions = result["predictions"]

        lines = [
            "AGRIASSIST-RX: IMAGE ANALYSIS",
            "",
            f"Image: {Path(image_path).name}",
            "",
            "Top 3 predictions:",
        ]

        for rank, item in enumerate(
            predictions, start=1
        ):

            condition = item["condition"]
            score = (
                item["model_score"] * 100
            )

            lines.append(
                f"{rank}. {condition}: {score:.2f}%"
            )

        lines.append("")

        if result["review_recommended"]:

            lines.append(
                "Review recommended: the top "
                "prediction has a low score "
                "or a small margin over the "
                "second-ranked class."
            )

        else:

            lines.append(
                "The model has a clear "
                "top-ranked class under the "
                "provisional screening rule."
            )

        lines.extend([
            "",
            "IMPORTANT:",
            "This is an experimental wheat "
            "condition classifier.",
            "A high model score does not "
            "confirm a diagnosis.",
            "The model cannot reliably "
            "reject unrelated or unclear images.",
            "Seek independent verification "
            "before making treatment decisions.",
        ])

        return {
            "route": "vision",
            "prediction": result,
            "response": "\n".join(lines),
            "error": "",
        }

    except (
        FileNotFoundError,
        ValueError,
        OSError,
        RuntimeError,
    ) as error:

        return {
            "route": "vision",
            "error": str(error),
            "response": (
                "Image analysis could not be "
                f"completed: {error}"
            ),
        }


# =====================================================
# 4. BUILD THE LANGGRAPH WORKFLOW
# =====================================================

def build_vision_graph():

    graph = StateGraph(VisionAgentState)

    graph.add_node(
        "vision_analysis",
        vision_node
    )

    graph.add_edge(
        START,
        "vision_analysis"
    )

    graph.add_edge(
        "vision_analysis",
        END
    )

    return graph.compile()


# =====================================================
# 5. REUSABLE FUNCTION FOR THE MAIN AGENT
# =====================================================

def run_vision_agent(
    image_path: str,
    question: str = "Analyze this wheat image",
):

    agent = build_vision_graph()

    initial_state = {
        "image_path": image_path,
        "question": question,
    }

    return agent.invoke(initial_state)


# =====================================================
# 6. COMMAND-LINE TEST
# =====================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run AgriAssist-RX Computer Vision "
            "through LangGraph."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Full path to the wheat photograph",
    )

    parser.add_argument(
        "--question",
        default="Analyze this wheat image",
        help="Optional text describing the request",
    )

    args = parser.parse_args()

    result = run_vision_agent(
        image_path=args.image,
        question=args.question,
    )

    print("\n" + "=" * 60)
    print("LANGGRAPH IMAGE ROUTE")
    print("=" * 60)

    print("Selected route:", result.get("route"))

    print("\n" + result.get(
        "response",
        "No response generated."
    ))

    if result.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
