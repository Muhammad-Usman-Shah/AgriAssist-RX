
"""
AgriAssist-RX
Phase 4: Five-Route LangGraph Agent

Available routes:
1. Agricultural RAG
2. Live weather
3. Agricultural calculator
4. CSV data analysis
5. Unsupported requests

CSV column matching has been corrected to:
- Ignore filenames when detecting columns
- Match specific column names before general aliases
- Handle fertilizer cost and wheat yield questions
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


# ==========================================
# STEP 1: Shared agent state
# ==========================================

class AgentState(TypedDict):
    question: str
    selected_tool: str
    calculation: str
    arg1: str | None
    arg2: str | None
    answer: str
    sources: list
    supported: bool


# ==========================================
# STEP 2: Choose the appropriate tool
# ==========================================

def choose_tool(state: AgentState):

    question = state["question"].lower()

    csv_indicators = [
        ".csv",
        "csv file",
        "csv data",
        "dataset",
    ]

    if any(
        indicator in question
        for indicator in csv_indicators
    ):
        return {"selected_tool": "csv"}

    # Use Gemini for other tool decisions.
    return gemini_router(state)


# ==========================================
# STEP 3: Identify the CSV column
# ==========================================

def identify_column(question, columns):

    # Remove the filename before searching
    # for column names.
    clean_question = re.sub(
        r"\b[A-Za-z0-9_-]+\.csv\b",
        " ",
        question.lower()
    )

    text = clean_question.replace("_", " ")

    # First, look for actual column names.
    # Check longer names first.
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

    # Next, match natural-language phrases
    # to available agricultural columns.
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


# ==========================================
# STEP 4: Analyze the CSV dataset
# ==========================================

def csv_node(state: AgentState):

    question = state["question"]
    question_lower = question.lower()

    # Extract CSV filename from the question.
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

        # ----------------------------------
        # Handle row-count questions
        # ----------------------------------

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

        # ----------------------------------
        # Identify the target column
        # ----------------------------------

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

        # ----------------------------------
        # Perform numerical analysis
        # ----------------------------------

        result = analyze_csv(
            filename,
            column
        )

        # ----------------------------------
        # Identify requested statistic
        # ----------------------------------

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

            # If no specific statistic is
            # requested, show all statistics.
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

        # ----------------------------------
        # Format the result with units
        # ----------------------------------

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


# ==========================================
# STEP 5: Build the LangGraph workflow
# ==========================================

workflow = StateGraph(AgentState)

workflow.add_node(
    "router",
    choose_tool
)

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

workflow.add_edge(
    START,
    "router"
)

workflow.add_conditional_edges(
    "router",
    lambda state: state["selected_tool"],
    {
        "rag": "rag",
        "weather": "weather",
        "calculator": "calculator",
        "csv": "csv",
        "unsupported": "unsupported",
    }
)

for node in [
    "rag",
    "weather",
    "calculator",
    "csv",
    "unsupported"
]:

    workflow.add_edge(
        node,
        END
    )

app = workflow.compile()


# ==========================================
# STEP 6: Interactive application
# ==========================================

def main():

    print("=" * 55)
    print("AGRIASSIST-RX: FIVE-ROUTE AI AGENT")
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

        # Close the Gemini client used by
        # the agricultural RAG module.
        if "rag" in sys.modules:

            from rag import client

            client.close()


if __name__ == "__main__":
    main()
