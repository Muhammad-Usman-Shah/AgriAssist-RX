
"""
AgriAssist-RX
Phase 6: Intelligent Agent + Agricultural Calculator

Four available routes:
1. Agricultural RAG
2. Live weather
3. Agricultural calculator
4. Unsupported requests

Calculator operations:
1. Fertilizer quantity
2. Fertilizer cost
3. Seed requirement
4. Area conversion
5. Irrigation volume
6. Crop production (NEW)

Reuses existing tested project functions.
"""

import os
import re
import sys

from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END

from calculator_tool import (
    fertilizer_quantity,
    fertilizer_cost,
    seed_requirement,
    acres_to_hectares,
    irrigation_volume,
)

from agent_intelligent import (
    rag_tool,
    weather_tool,
)


# ==========================================
# STEP 1: Configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.1-flash-lite"


# ==========================================
# STEP 2: Structured AI decisions
# ==========================================

class ToolName(str, Enum):
    RAG = "rag"
    WEATHER = "weather"
    CALCULATOR = "calculator"
    UNSUPPORTED = "unsupported"


class CalculationName(str, Enum):
    NONE = "none"
    FERTILIZER_QUANTITY = "fertilizer_quantity"
    FERTILIZER_COST = "fertilizer_cost"
    SEED_REQUIREMENT = "seed_requirement"
    AREA_CONVERSION = "area_conversion"
    IRRIGATION_VOLUME = "irrigation_volume"
    YIELD_PRODUCTION = "yield_production"


class AgentDecision(BaseModel):
    tool: ToolName
    calculation: CalculationName
    arg1: str | None
    arg2: str | None


# ==========================================
# STEP 3: Shared graph state
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
# STEP 4: Gemini router and input extraction
# ==========================================

def choose_tool(state: AgentState):

    # Deterministic extraction for crop production
    # from an explicitly stated yield rate and area.
    # This does not require a Gemini API call.

    question = state["question"]
    q = question.lower()

    numeric = r"(\d+(?:,\d{3})*(?:\.\d+)?)"

    rate_match = re.search(
        numeric
        + r"\s*(?:kg|kilograms?)\s*(?:/|per)\s*acre\b",
        q,
    )

    area_match = re.search(
        numeric + r"\s*acres?\b",
        q,
    )

    production_intent = (
        "production" in q
        or "total yield" in q
        or "harvest" in q
    )

    if rate_match and area_match and production_intent:

        return {
            "selected_tool": "calculator",
            "calculation": "yield_production",
            "arg1": area_match.group(1).replace(",", ""),
            "arg2": rate_match.group(1).replace(",", ""),
        }

    # Existing Gemini routing for other requests.

    if not API_KEY:

        return {
            "selected_tool": "unsupported",
            "answer": "Gemini API key is missing."
        }

    instruction = """
You are the tool router for AgriAssist-RX.

Choose exactly one tool:

RAG:
Agricultural knowledge, wheat diseases,
crop explanations, and effects of weather
on crop diseases.

WEATHER:
Actual current weather information for
Pakistani cities. Historical weather and
long-range forecasts are not supported.

CALCULATOR:
Arithmetic using user-provided numeric
values and the following operations:

fertilizer_quantity:
arg1 = area in acres
arg2 = fertilizer rate in bags per acre

fertilizer_cost:
arg1 = number of bags
arg2 = price per bag in PKR

seed_requirement:
arg1 = area in acres
arg2 = seed rate in kg per acre

area_conversion:
arg1 = area in acres
arg2 = null

irrigation_volume:
arg1 = area in acres
arg2 = water depth in millimetres

yield_production:
arg1 = field area in acres
arg2 = wheat/crop yield rate in kg per acre
Calculate total production in kg. Use this operation
when the user provides both values and asks for
production, harvest quantity, or total yield.

If a calculation is requested but required
numeric values are missing, select CALCULATOR
and set the missing argument to null.

Only extract numeric values explicitly
provided by the user. Do not estimate,
invent, or calculate any values yourself.

Respect units. If a value is supplied in
an incompatible unit, set that argument
to null instead of converting it.

If the requested operation is unknown or
unsupported, choose UNSUPPORTED.

UNSUPPORTED:
Live market prices, live news, missing
images, external policy announcements,
legal pesticide dosage recommendations,
or questions requiring unavailable data.

If the user asks for a calculation that
requires a current market price or another
external value not provided, choose
UNSUPPORTED.

For non-calculator tools, use:
calculation = none
arg1 = null
arg2 = null

Return structured data only.

User question:
""" + state["question"]

    client = None

    try:

        client = genai.Client(api_key=API_KEY)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=instruction,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AgentDecision,
                temperature=0,
            ),
        )

        if response.parsed is not None:

            decision = AgentDecision.model_validate(
                response.parsed
            )

        else:

            decision = AgentDecision.model_validate_json(
                response.text
            )

        return {
            "selected_tool": decision.tool.value,
            "calculation": decision.calculation.value,
            "arg1": decision.arg1,
            "arg2": decision.arg2,
        }

    except Exception as error:

        return {
            "selected_tool": "unsupported",
            "answer": (
                "The AI router is temporarily "
                "unavailable. Try again later. "
                f"Technical details: {error}"
            ),
        }

    finally:

        if client is not None:
            client.close()


# ==========================================
# STEP 5: Validate extracted values
# ==========================================

def numbers_in_question(question):

    pattern = (
        r"(?<![A-Za-z0-9])"
        r"\d+(?:,\d{3})*(?:\.\d+)?"
        r"(?![A-Za-z0-9])"
    )

    values = set()

    for text in re.findall(pattern, question):

        try:

            values.add(
                Decimal(text.replace(",", ""))
            )

        except InvalidOperation:
            continue

    return values


def verified_argument(value, question):

    if value is None:
        return None

    try:

        number = Decimal(
            str(value).replace(",", "").strip()
        )

    except (InvalidOperation, ValueError):
        return None

    if not number.is_finite():
        return None

    # Prevent use of numbers that were not
    # explicitly provided in the question.

    if number not in numbers_in_question(question):
        return None

    return str(number)


# ==========================================
# STEP 6: Calculator node
# ==========================================

def calculator_node(state: AgentState):

    question = state["question"]
    operation = state["calculation"]

    arg1 = verified_argument(
        state["arg1"],
        question
    )

    arg2 = verified_argument(
        state["arg2"],
        question
    )

    if operation == "area_conversion":
        required_inputs = 1

    else:
        required_inputs = 2

    missing = (
        arg1 is None
        or (
            required_inputs == 2
            and arg2 is None
        )
    )

    if missing:

        return {
            "answer": (
                "I need the required numerical "
                "inputs and their units to perform "
                "this calculation. Please provide "
                "them explicitly."
            ),
            "sources": [],
            "supported": False,
        }

    try:

        # --------------------------------------
        # Fertilizer quantity
        # --------------------------------------

        if operation == "fertilizer_quantity":

            result = fertilizer_quantity(
                arg1,
                arg2
            )

            answer = (
                "Fertilizer required: "
                f"{result['total_bags']} bags.\n"
                f"Inputs: {arg1} acres at "
                f"{arg2} bags per acre."
            )

        # --------------------------------------
        # Fertilizer cost
        # --------------------------------------

        elif operation == "fertilizer_cost":

            result = fertilizer_cost(
                arg1,
                arg2
            )

            answer = (
                "Total fertilizer cost: PKR "
                f"{result['total_cost_pkr']:,.2f}\n"
                f"Inputs: {arg1} bags at "
                f"PKR {arg2} per bag."
            )

        # --------------------------------------
        # Seed requirement
        # --------------------------------------

        elif operation == "seed_requirement":

            result = seed_requirement(
                arg1,
                arg2
            )

            answer = (
                "Total seed required: "
                f"{result['total_seed_kg']} kg.\n"
                f"Inputs: {arg1} acres at "
                f"{arg2} kg per acre."
            )

        # --------------------------------------
        # Area conversion
        # --------------------------------------

        elif operation == "area_conversion":

            result = acres_to_hectares(arg1)

            answer = (
                f"{arg1} acres = "
                f"{result['hectares']:.4f} hectares."
            )

        # --------------------------------------
        # Irrigation volume
        # --------------------------------------

        elif operation == "irrigation_volume":

            result = irrigation_volume(
                arg1,
                arg2
            )

            answer = (
                "Theoretical irrigation volume: "
                f"{result['volume_m3']:.2f} m³.\n"
                f"Inputs: {arg1} acres with "
                f"{arg2} mm applied depth.\n"
                "Excludes irrigation losses."
            )

        # --------------------------------------
        # NEW: Crop production
        # --------------------------------------

        elif operation == "yield_production":

            area_acres = Decimal(arg1)
            yield_kg_per_acre = Decimal(arg2)

            if area_acres <= 0:

                raise ValueError(
                    "Area must be positive."
                )

            if yield_kg_per_acre < 0:

                raise ValueError(
                    "Yield rate cannot be negative."
                )

            total_kg = (
                area_acres * yield_kg_per_acre
            )

            answer = (
                "Total crop production: "
                f"{total_kg:,.2f} kg.\n"
                f"Inputs: {arg1} acres at "
                f"{arg2} kg per acre."
            )

        # --------------------------------------
        # Unsupported calculation
        # --------------------------------------

        else:

            return {
                "answer": (
                    "This calculation is not "
                    "supported yet."
                ),
                "sources": [],
                "supported": False,
            }

        return {
            "answer": answer,
            "sources": [],
            "supported": True,
        }

    except (ValueError, ArithmeticError) as error:

        return {
            "answer": (
                f"Calculation error: {error}"
            ),
            "sources": [],
            "supported": False,
        }


# ==========================================
# STEP 7: Unsupported handler
# ==========================================

def unsupported_node(state: AgentState):

    return {
        "answer": (
            state.get("answer")
            or "I cannot fulfill this request with "
               "the currently available tools."
        ),
        "sources": [],
        "supported": False,
    }


# ==========================================
# STEP 8: Build LangGraph
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
        "unsupported": "unsupported",
    },
)

for node in [
    "rag",
    "weather",
    "calculator",
    "unsupported",
]:

    workflow.add_edge(
        node,
        END
    )

app = workflow.compile()


# ==========================================
# STEP 9: Interactive application
# ==========================================

def main():

    print("=" * 55)

    print(
        "AGRIASSIST-RX: FOUR-TOOL LANGGRAPH AGENT"
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

        if "rag" in sys.modules:

            from rag import client
            client.close()


if __name__ == "__main__":
    main()
