
"""
AgriAssist-RX
Phase 6: Improved Six-Route LangGraph Agent

Routes:
1. Agricultural RAG
2. Live weather
3. Agricultural calculator
4. CSV data analysis
5. Wheat image classification
6. Unsupported requests

Update:
- More flexible Computer Vision intent detection.
- Avoid confusing image-analysis requests with
  general educational questions about images.
- Preserve existing RAG, weather, calculator,
  CSV and unsupported functionality.
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
# STEP 2: Extract local image paths
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
# STEP 3: Detect Computer Vision requests
# ==================================================

def is_vision_request(question: str) -> bool:
    """
    Detect requests to analyze an actual image.

    Does not require the exact wording to match
    a hard-coded phrase.

    General educational questions about
    computer vision should not automatically
    use the Vision tool.
    """

    text = question.lower().strip()

    # If a local image path is supplied,
    # treat the request as image analysis.

    if extract_image_path(question):
        return True

    # General learning questions usually do not
    # request analysis of a particular image.

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

    # Look for image-related words.

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

    # Common action words used when a person
    # wants the agent to inspect an image.

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

    # Agriculture-related context helps identify
    # disease and crop-image analysis.

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

    # Detect whether the question refers
    # to an actual image being provided.

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

    # Example:
    # "Please examine this wheat photograph."

    if (
        contains_image_term
        and contains_action
        and refers_to_particular_image
    ):
        return True

    # Example:
    # "Run image classification on a wheat
    # plant photo."
    #
    # This route is also useful when the image
    # has not yet been supplied.

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

    # Example:
    # "What disease is visible in this
    # wheat image?"

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
# STEP 4: Select the appropriate tool
# ==================================================

def choose_tool(state: AgentState):

    question = state["question"]

    question_lower = question.lower()

    # ----------------------------------------------
    # ROUTE 1: COMPUTER VISION
    # ----------------------------------------------

    if is_vision_request(question):

        return {
            "selected_tool": "vision",
            "image_path": extract_image_path(
                question
            ),
        }

    # ----------------------------------------------
    # ROUTE 2: CSV ANALYSIS
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
    # ROUTE 3: LIVE WEATHER
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
    # OTHER ROUTES
    # ----------------------------------------------

    # Gemini handles:
    # - Agricultural RAG
    # - Calculator
    # - Unsupported requests
    #
    # The calculator's yield_production operation
    # is implemented in agent_calculator.py.

    return gemini_router(state)


# ==================================================
# STEP 5: Computer Vision node
# ==================================================

def vision_node(state: AgentState):

    image_path = (
        state.get("image_path", "")
        or extract_image_path(state["question"])
    )

    # Reuse existing LangGraph Vision node.

    result = existing_vision_node({
        "image_path": image_path,
        "question": state["question"],
    })

    prediction = result.get(
        "prediction", {}
    )

    error = result.get(
        "error", ""
    )

    return {
        "selected_tool": "vision",
        "image_path": image_path,
        "answer": result.get(
            "response",
            "No image analysis returned."
        ),
        "sources": [],
        "supported": (
            bool(prediction)
            and not bool(error)
        ),
        "vision_prediction": prediction,
    }


# ==================================================
# STEP 6: Identify CSV columns
# ==================================================

def identify_column(question, columns):

    # Remove the filename before looking for
    # CSV column names.

    clean_question = re.sub(
        r"\b[A-Za-z0-9_-]+\.csv\b",
        " ",
        question.lower()
    )

    text = clean_question.replace("_", " ")

    # Match actual column names first.
    # Match longer column names before shorter.

    for column in sorted(
        columns,
        key=len,
        reverse=True
    ):

        readable = (
            column.lower().replace("_", " ")
        )

        pattern = (
            r"\b"
            + re.escape(readable)
            + r"\b"
        )

        if re.search(pattern, text):
            return column

    # Natural-language aliases.

    aliases = [
        (
            "fertilizer cost",
            "fertilizer_cost_pkr"
        ),
        (
            "fertilizer expense",
            "fertilizer_cost_pkr"
        ),
        (
            "fertilizer spending",
            "fertilizer_cost_pkr"
        ),
        (
            "production",
            "yield_kg"
        ),
        (
            "yield",
            "yield_kg"
        ),
        (
            "field area",
            "area_acres"
        ),
        (
            "cultivated area",
            "area_acres"
        ),
        (
            "acreage",
            "area_acres"
        ),
    ]

    for phrase, column in aliases:

        if column not in columns:
            continue

        pattern = (
            r"\b"
            + re.escape(phrase)
            + r"\b"
        )

        if re.search(pattern, text):
            return column

    return None


# ==================================================
# STEP 7: CSV analysis node
# ==================================================

def csv_node(state: AgentState):

    question = state["question"]

    question_lower = question.lower()

    # Extract CSV filename.

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

        # ------------------------------------------
        # Row-count questions
        # ------------------------------------------

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

        # ------------------------------------------
        # Identify target column
        # ------------------------------------------

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

        # ------------------------------------------
        # Calculate numerical statistics
        # ------------------------------------------

        result = analyze_csv(
            filename,
            column
        )

        # ------------------------------------------
        # Identify requested statistic
        # ------------------------------------------

        if any(
            word in question_lower
            for word in ["average", "mean"]
        ):

            metric = "AVERAGE"
            value = result["average"]

        elif any(
            word in question_lower
            for word in [
                "minimum",
                "lowest",
                "min"
            ]
        ):

            metric = "MINIMUM"
            value = result["minimum"]

        elif any(
            word in question_lower
            for word in [
                "maximum",
                "highest",
                "max"
            ]
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

            # Return all statistics when the
            # requested statistic is unclear.

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

        # ------------------------------------------
        # Format value with units
        # ------------------------------------------

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
# STEP 8: Build LangGraph workflow
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

# Start from the router.

workflow.add_edge(
    START,
    "router"
)

# Route according to the selected tool.

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
    }
)

# Every route ends after its tool executes.

for node in [
    "rag",
    "weather",
    "calculator",
    "csv",
    "vision",
    "unsupported",
]:

    workflow.add_edge(
        node,
        END
    )

# Compile LangGraph.

app = workflow.compile()


# ==================================================
# STEP 9: Interactive application
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

                # Show RAG evidence sources.

                if result["sources"]:

                    print("\nEVIDENCE SOURCES:")

                    for source in result["sources"]:

                        print(
                            f"[{source['id']}] "
                            f"{source['source']} "
                            f"(PDF page {source['page']})"
                        )

                print(
                    "\n" + "-" * 55 + "\n"
                )

            except Exception as error:

                print(
                    "\nUnable to process request."
                )

                print(
                    "Error:",
                    error
                )

    finally:

        # Close Gemini client if RAG was loaded.

        if "rag" in sys.modules:

            from rag import client
            client.close()


if __name__ == "__main__":
    main()
