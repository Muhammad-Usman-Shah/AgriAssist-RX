
"""
AgriAssist-RX
Phase 5: Six-Route LangGraph Agent

Routes:
1. Agricultural RAG
2. Live weather
3. Agricultural calculator
4. CSV data analysis
5. Wheat image classification
6. Unsupported requests

Weather, CSV and image routing use local rules.
Other requests use the existing Gemini router.
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

from agent_vision import vision_node as existing_vision_node


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
# STEP 2: Extract image path
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
# STEP 3: Select the appropriate tool
# ==================================================

def choose_tool(state: AgentState):

    question = state["question"]
    question_lower = question.lower()

    # ----------------------------------------------
    # ROUTE 1: COMPUTER VISION
    # ----------------------------------------------

    image_path = extract_image_path(question)

    if image_path:

        return {
            "selected_tool": "vision",
            "image_path": image_path,
        }

    vision_phrases = [
        "analyze this image",
        "analyse this image",
        "analyze this photo",
        "analyse this photo",
        "analyze wheat image",
        "analyse wheat image",
        "classify this image",
        "classify this photo",
        "classify wheat image",
        "predict from image",
        "identify disease in image",
        "identify disease from photo",
    ]

    if any(
        phrase in question_lower
        for phrase in vision_phrases
    ):

        return {
            "selected_tool": "vision",
            "image_path": "",
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
    # ROUTE 3: WEATHER
    # ----------------------------------------------
    # Weather requests do not need Gemini routing.
    # This avoids a routing failure when Gemini
    # returns a temporary 503 UNAVAILABLE error.

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
    # OTHER ROUTES: EXISTING GEMINI ROUTER
    # ----------------------------------------------
    # Handles RAG, calculator and unsupported
    # requests using the existing implementation.

    return gemini_router(state)


# ==================================================
# STEP 4: Computer Vision node
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
        "supported": bool(prediction) and not bool(error),
        "vision_prediction": prediction,
    }


# ==================================================
# STEP 5: Identify CSV columns
# ==================================================

def identify_column(question, columns):

    # Remove CSV filename from the question.

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

    # Natural-language aliases.

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
# STEP 6: CSV analysis node
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

        # ------------------------------------------
        # Handle row-count questions
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
        # Analyze numerical data
        # ------------------------------------------

        result = analyze_csv(
            filename,
            column
        )

        # ------------------------------------------
        # Select requested statistic
        # ------------------------------------------

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

        # ------------------------------------------
        # Format value with appropriate units
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
# STEP 7: Build LangGraph workflow
# ==================================================

workflow = StateGraph(AgentState)

# Router

workflow.add_node(
    "router",
    choose_tool
)

# Existing tools

workflow.add_node(
    "rag",
    rag_tool
)

workflow.add_node(
    "weather",
    weather_tool
)

workflow.add_node(
    "calculator",
    calculator_node
)

workflow.add_node(
    "csv",
    csv_node
)

workflow.add_node(
    "unsupported",
    unsupported_node
)

# Computer Vision tool

workflow.add_node(
    "vision",
    vision_node
)

# Entry point

workflow.add_edge(
    START,
    "router"
)

# Conditional routing

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

# End all tool routes.

for node in [
    "rag",
    "weather",
    "calculator",
    "csv",
    "vision",
    "unsupported"
]:

    workflow.add_edge(
        node,
        END
    )

# Compile agent.

app = workflow.compile()


# ==================================================
# STEP 8: Interactive application
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

                if result["sources"]:

                    print("\nEVIDENCE SOURCES:")

                    for source in result["sources"]:

                        print(
                            f"[{source['id']}] "
                            f"{source['source']} "
                            f"(PDF page {source['page']})"
                        )

                print("\n" + "-" * 55 + "\n")

            except Exception as error:

                print(
                    "\nUnable to process request."
                )

                print(
                    "Error:",
                    error
                )

    finally:

        # Close Gemini RAG client if loaded.

        if "rag" in sys.modules:

            from rag import client
            client.close()


if __name__ == "__main__":
    main()
