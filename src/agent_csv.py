
"""
AgriAssist-RX
Phase 6: Six-Route LangGraph Agent

Supported routes:
1. Agricultural RAG
2. Live weather
3. Agricultural calculator
4. CSV data analysis
5. Wheat image classification
6. Unsupported requests

Additional system state:
- api_error: temporary Gemini routing failure

Updates:
- Improved image intent detection
- Direct weather and CSV routing
- Separate API errors from unsupported requests
- Preserve existing calculator operations
- Preserve CSV statistics and RAG citations
"""

import re
import sys
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from agent_calculator import (
    choose_tool as gemini_router,
    rag_tool,
    weather_tool,
    calculator_node,
    unsupported_node,
)

from csv_tool import load_csv, analyze_csv

from agent_vision import (
    vision_node as existing_vision_node,
)


# ==================================================
# STEP 1: Shared agent state
# ==================================================

class AgentState(TypedDict):
    question: str
    selected_tool: str
    calculation: str
    arg1: str | None
    arg2: str | None
    answer: str
    sources: list
    supported: bool
    image_path: str
    vision_prediction: dict


# ==================================================
# STEP 2: Extract Windows image paths
# ==================================================

IMAGE_PATH_RE = re.compile(
    r"(?i)([A-Z]:[\\/][^\r\n\"<>|?*]*?\."
    r"(?:jpg|jpeg|png|webp|bmp))"
)


def extract_image_path(question: str) -> str:

    match = IMAGE_PATH_RE.search(question)

    if match:
        return match.group(1).strip()

    return ""


# ==================================================
# STEP 3: Recognize image-analysis requests
# ==================================================

def is_vision_request(question: str) -> bool:

    text = question.lower().strip()

    # An image path is a strong routing indicator.

    if extract_image_path(question):
        return True

    # Avoid routing general educational
    # questions to the image classifier.

    educational_patterns = [
        r"^\s*what\s+is\s+image\s+classification\b",
        r"^\s*what\s+are\s+image\s+classification\b",
        r"^\s*explain\s+how\s+image\s+classification\b",
        r"^\s*how\s+does\s+image\s+classification\b",
        r"^\s*how\s+do\s+image\s+classifiers\b",
        r"^\s*what\s+is\s+computer\s+vision\b",
        r"^\s*explain\s+computer\s+vision\b",
        r"^\s*how\s+does\s+computer\s+vision\b",
    ]

    for pattern in educational_patterns:
        if re.search(pattern, text):
            return False

    image_terms = [
        "image",
        "photo",
        "photograph",
        "picture",
        "snapshot",
    ]

    contains_image_term = any(
        re.search(
            r"\b" + re.escape(term) + r"s?\b",
            text
        )
        for term in image_terms
    )

    if not contains_image_term:
        return False

    action_patterns = [
        r"\banaly[sz]e\b",
        r"\bexamine\b",
        r"\bclassify\b",
        r"\bclassification\b",
        r"\bidentify\b",
        r"\bdetect\b",
        r"\bpredict\b",
        r"\brecognize\b",
        r"\brecognise\b",
        r"\binspect\b",
        r"\bcheck\b",
        r"\bdiagnos(?:e|is)\b",
        r"\bscan\b",
    ]

    contains_action = any(
        re.search(pattern, text)
        for pattern in action_patterns
    )

    agricultural_patterns = [
        r"\bwheat\b",
        r"\bcrop\b",
        r"\bplant\b",
        r"\bleaf\b",
        r"\bleaves\b",
        r"\bdisease\b",
        r"\brust\b",
        r"\bpest\b",
        r"\baphid\b",
    ]

    contains_agriculture = any(
        re.search(pattern, text)
        for pattern in agricultural_patterns
    )

    particular_image_patterns = [
        r"\bthis\b",
        r"\bthat\b",
        r"\bthese\b",
        r"\bmy\b",
        r"\battached\b",
        r"\buploaded\b",
        r"\bshown\b",
        r"\bvisible\b",
    ]

    refers_to_particular_image = any(
        re.search(pattern, text)
        for pattern in particular_image_patterns
    )

    if (
        contains_image_term
        and contains_action
        and refers_to_particular_image
    ):
        return True

    explicit_vision_patterns = [
        r"\brun\s+image\s+classification\b",
        r"\bperform\s+image\s+classification\b",
        r"\buse\s+image\s+classification\b",
        r"\bclassify\s+(?:the\s+)?wheat\b",
        r"\bpredict\s+(?:the\s+)?disease\s+from\b",
        r"\bidentify\s+(?:the\s+)?condition\b",
    ]

    if contains_agriculture:
        for pattern in explicit_vision_patterns:
            if re.search(pattern, text):
                return True

    visual_diagnosis_patterns = [
        r"\bwhat\s+disease\s+is\s+visible\b",
        r"\bwhat\s+condition\s+is\s+visible\b",
        r"\bwhich\s+disease\s+is\s+shown\b",
        r"\bwhat\s+disease\s+does\s+this\b",
    ]

    if contains_agriculture:
        for pattern in visual_diagnosis_patterns:
            if re.search(pattern, text):
                return True

    return False


# ==================================================
# STEP 4: Identify Gemini routing errors
# ==================================================

API_ERROR_INDICATORS = [
    "router is temporarily unavailable",
    "gemini api key is missing",
    "resource_exhausted",
    "quota exceeded",
    "rate limit exceeded",
    "503 unavailable",
    "429 resource_exhausted",
    "api key not valid",
    "authentication failed",
]


def is_api_routing_error(result: dict) -> bool:
    """
    Detect a failure returned by the existing
    Gemini router.

    The original calculator module sometimes
    returns 'unsupported' when an API call fails.
    We distinguish that from a genuine
    unsupported user request.
    """

    if not isinstance(result, dict):
        return False

    answer = str(result.get("answer") or "").lower()

    return any(
        indicator in answer
        for indicator in API_ERROR_INDICATORS
    )


# ==================================================
# STEP 5: Main router
# ==================================================

def choose_tool(state: AgentState):

    question = state["question"]
    question_lower = question.lower()

    # ----------------------------------------------
    # ROUTE 1: Computer Vision
    # ----------------------------------------------

    if is_vision_request(question):

        return {
            "selected_tool": "vision",
            "image_path": extract_image_path(question),
        }

    # ----------------------------------------------
    # ROUTE 2: CSV
    # ----------------------------------------------

    csv_indicators = [
        ".csv",
        "csv file",
        "csv data",
        "dataset",
    ]

    if any(
        indicator in question_lower
        for indicator in csv_indicators
    ):

        return {
            "selected_tool": "csv",
        }

    # ----------------------------------------------
    # ROUTE 3: Weather
    # ----------------------------------------------

    weather_phrases = [
        "weather",
        "temperature",
        "forecast",
        "rainfall",
        "humidity",
        "wind speed",
    ]

    if any(
        phrase in question_lower
        for phrase in weather_phrases
    ):

        return {
            "selected_tool": "weather",
        }

    # ----------------------------------------------
    # ROUTES 4-6: Existing Gemini router
    # ----------------------------------------------

    # The Gemini router selects agricultural RAG,
    # calculator, or unsupported requests.
    #
    # It also extracts calculator arguments.

    try:

        result = gemini_router(state)

    except Exception:

        return {
            "selected_tool": "api_error",
            "answer": (
                "The AI routing service is temporarily "
                "unavailable or its request limit has "
                "been reached. Please try this request "
                "again later."
            ),
            "sources": [],
            "supported": False,
        }

    if is_api_routing_error(result):

        return {
            "selected_tool": "api_error",
            "answer": (
                "The AI routing service is temporarily "
                "unavailable or its request limit has "
                "been reached. Please try this request "
                "again later."
            ),
            "sources": [],
            "supported": False,
        }

    return result


# ==================================================
# STEP 6: API error response node
# ==================================================

def api_error_node(state: AgentState):
    """
    Return a clear error message without
    pretending that the request was unsupported.
    """

    return {
        "selected_tool": "api_error",
        "answer": (
            state.get("answer")
            or (
                "The AI routing service is temporarily "
                "unavailable or its request limit has "
                "been reached. Please try this request "
                "again later."
            )
        ),
        "sources": [],
        "supported": False,
    }


# ==================================================
# STEP 7: Computer Vision node
# ==================================================

def vision_node(state: AgentState):

    image_path = (
        state.get("image_path", "")
        or extract_image_path(state["question"])
    )

    result = existing_vision_node({
        "image_path": image_path,
        "question": state["question"],
    })

    prediction = result.get("prediction", {})
    error = result.get("error", "")

    return {
        "selected_tool": "vision",
        "image_path": image_path,
        "answer": result.get(
            "response",
            "No image analysis returned."
        ),
        "sources": [],
        "supported": (
            bool(prediction) and not bool(error)
        ),
        "vision_prediction": prediction,
    }


# ==================================================
# STEP 8: Identify CSV column
# ==================================================

def identify_column(question, columns):

    clean_question = re.sub(
        r"\b[A-Za-z0-9_-]+\.csv\b",
        " ",
        question.lower()
    )

    text = clean_question.replace("_", " ")

    # Match actual column names first.

    for column in sorted(
        columns,
        key=len,
        reverse=True
    ):

        readable = column.lower().replace("_", " ")

        pattern = (
            r"\b" + re.escape(readable) + r"\b"
        )

        if re.search(pattern, text):
            return column

    # Common agricultural column aliases.

    aliases = [
        ("fertilizer cost", "fertilizer_cost_pkr"),
        ("fertilizer expense", "fertilizer_cost_pkr"),
        ("fertilizer spending", "fertilizer_cost_pkr"),
        ("production", "yield_kg"),
        ("yield", "yield_kg"),
        ("field area", "area_acres"),
        ("cultivated area", "area_acres"),
        ("acreage", "area_acres"),
    ]

    for phrase, column in aliases:

        if column not in columns:
            continue

        pattern = (
            r"\b" + re.escape(phrase) + r"\b"
        )

        if re.search(pattern, text):
            return column

    return None


# ==================================================
# STEP 9: CSV analysis node
# ==================================================

def csv_node(state: AgentState):

    question = state["question"]
    question_lower = question.lower()

    filename_match = re.search(
        r"\b[A-Za-z0-9_-]+\.csv\b",
        question,
        flags=re.IGNORECASE
    )

    if not filename_match:

        return {
            "answer": (
                "Please specify a CSV filename. "
                "Example: What is the average yield "
                "in crop_production.csv?"
            ),
            "sources": [],
            "supported": False,
        }

    filename = filename_match.group(0)

    try:

        columns, rows = load_csv(filename)

        row_phrases = [
            "how many rows",
            "number of rows",
            "row count",
            "how many records",
            "number of records",
        ]

        if any(
            phrase in question_lower
            for phrase in row_phrases
        ):

            return {
                "answer": (
                    f"Dataset: {filename}\n"
                    f"Total data rows: {len(rows)}"
                ),
                "sources": [],
                "supported": True,
            }

        column = identify_column(
            question,
            columns
        )

        if column is None:

            return {
                "answer": (
                    "I could not identify the column "
                    "you want to analyze.\n"
                    "Available columns: "
                    + ", ".join(columns)
                ),
                "sources": [],
                "supported": False,
            }

        result = analyze_csv(
            filename,
            column
        )

        # Determine requested statistic.

        if any(
            word in question_lower
            for word in ["average", "mean"]
        ):

            metric = "AVERAGE"
            value = result["average"]

        elif any(
            word in question_lower
            for word in ["minimum", "lowest", "min"]
        ):

            metric = "MINIMUM"
            value = result["minimum"]

        elif any(
            word in question_lower
            for word in ["maximum", "highest", "max"]
        ):

            metric = "MAXIMUM"
            value = result["maximum"]

        elif any(
            word in question_lower
            for word in ["total", "sum"]
        ):

            metric = "SUM"
            value = result["sum"]

        else:

            return {
                "answer": (
                    f"Dataset: {filename}\n"
                    f"Column: {column}\n"
                    f"Rows: {result['total_rows']}\n"
                    f"Sum: {result['sum']:,.2f}\n"
                    f"Average: {result['average']:,.2f}\n"
                    f"Minimum: {result['minimum']:,.2f}\n"
                    f"Maximum: {result['maximum']:,.2f}"
                ),
                "sources": [],
                "supported": True,
            }

        # Format with units.

        formatted_value = f"{value:,.2f}"

        if column.endswith("_pkr"):

            formatted_value = (
                "PKR " + formatted_value
            )

        elif column.endswith("_kg"):

            formatted_value += " kg"

        elif column.endswith("_acres"):

            formatted_value += " acres"

        answer = (
            f"Dataset: {filename}\n"
            f"Column: {column}\n"
            f"{metric}: {formatted_value}\n"
            f"Analyzed {result['total_rows']} rows."
        )

        return {
            "answer": answer,
            "sources": [],
            "supported": True,
        }

    except (
        ValueError,
        FileNotFoundError,
        OSError
    ) as error:

        return {
            "answer": (
                f"CSV analysis error: {error}"
            ),
            "sources": [],
            "supported": False,
        }


# ==================================================
# STEP 10: Build LangGraph workflow
# ==================================================

workflow = StateGraph(AgentState)

# Router

workflow.add_node(
    "router",
    choose_tool
)

# Agricultural RAG

workflow.add_node(
    "rag",
    rag_tool
)

# Live weather

workflow.add_node(
    "weather",
    weather_tool
)

# Agricultural calculator

workflow.add_node(
    "calculator",
    calculator_node
)

# CSV analysis

workflow.add_node(
    "csv",
    csv_node
)

# Computer Vision

workflow.add_node(
    "vision",
    vision_node
)

# Unsupported requests

workflow.add_node(
    "unsupported",
    unsupported_node
)

# API failures

workflow.add_node(
    "api_error",
    api_error_node
)

# Start graph

workflow.add_edge(
    START,
    "router"
)

# Route according to tool selection.

workflow.add_conditional_edges(
    "router",
    lambda state: state["selected_tool"],
    {
        "rag": "rag",
        "weather": "weather",
        "calculator": "calculator",
        "csv": "csv",
        "vision": "vision",
        "unsupported": "unsupported",
        "api_error": "api_error",
    }
)

# Each node ends its workflow.

for node in [
    "rag",
    "weather",
    "calculator",
    "csv",
    "vision",
    "unsupported",
    "api_error",
]:

    workflow.add_edge(
        node,
        END
    )

app = workflow.compile()


# ==================================================
# STEP 11: Interactive application
# ==================================================

def main():

    print("=" * 55)
    print(
        "AGRIASSIST-RX: SIX-ROUTE AI AGENT"
    )
    print("=" * 55)

    print("\nType 'exit' to stop.\n")

    try:

        while True:

            question = input(
                "Your question: "
            ).strip()

            if question.lower() in (
                "exit",
                "quit"
            ):

                print("\nGoodbye!")
                break

            if not question:
                continue

            try:

                result = app.invoke({
                    "question": question,
                    "selected_tool": "",
                    "calculation": "none",
                    "arg1": None,
                    "arg2": None,
                    "answer": "",
                    "sources": [],
                    "supported": False,
                    "image_path": "",
                    "vision_prediction": {},
                })

                print("\n" + "-" * 55)

                print(
                    "Selected tool:",
                    result["selected_tool"]
                )

                print("-" * 55)

                print("\nAGRIASSIST-RX ANSWER:")
                print(result["answer"])

                if result.get("sources"):

                    print("\nEVIDENCE SOURCES:")

                    for source in result["sources"]:

                        print(
                            f"[{source['id']}] "
                            f"{source['source']} "
                            f"(PDF page {source['page']})"
                        )

                print("\n" + "-" * 55 + "\n")

            except Exception as error:

                print("\nUnable to process request.")
                print("Error:", error)

    finally:

        if "rag" in sys.modules:

            from rag import client
            client.close()


if __name__ == "__main__":
    main()
