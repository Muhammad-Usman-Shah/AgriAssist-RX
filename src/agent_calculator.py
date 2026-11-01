"""AgriAssist-RX: Gemini tool router and agricultural calculator.

This module is imported by the six-route agent in src/agent_csv.py.
It also supports the original standalone four-route LangGraph workflow.
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
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from calculator_tool import (
    acres_to_hectares,
    fertilizer_cost,
    fertilizer_quantity,
    irrigation_volume,
    seed_requirement,
)
from agent_intelligent import rag_tool, weather_tool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"


class ToolName(str, Enum):
    RAG = "rag"
    WEATHER = "weather"
    CALCULATOR = "calculator"
    UNSUPPORTED = "unsupported"


class CalculationName(str, Enum):
    NONE = "none"
    FERTILIZER_QUANTITY = "fertilizer_quantity"
    FERTILIZER_COST = "fertilizer_cost"
    FERTILIZER_AREA_COST = "fertilizer_area_cost"
    SEED_REQUIREMENT = "seed_requirement"
    AREA_CONVERSION = "area_conversion"
    IRRIGATION_VOLUME = "irrigation_volume"
    YIELD_PRODUCTION = "yield_production"


class AgentDecision(BaseModel):
    tool: ToolName
    calculation: CalculationName
    arg1: str | None
    arg2: str | None


class AgentState(TypedDict):
    question: str
    selected_tool: str
    calculation: str
    arg1: str | None
    arg2: str | None
    answer: str
    sources: list
    supported: bool


# A numeric value optionally containing a sign, comma separators, or decimals.
NUMBER = r"(?<![A-Za-z0-9_.])([-+]?\d+(?:,\d{3})*(?:\.\d+)?)"


def find_area_acres(question: str) -> str | None:
    match = re.search(NUMBER + r"\s*acres?\b", question, re.I)
    return match.group(1).replace(",", "") if match else None


def find_pkr_per_acre(question: str) -> str | None:
    """Find rates expressed as '3000 PKR per acre' or 'PKR 3000/acre'."""
    after = re.search(
        NUMBER + r"\s*(?:pkr|rs\.?)\s*(?:/|per)\s*acre\b",
        question,
        re.I,
    )
    before = re.search(
        r"\b(?:pkr|rs\.?)\s*" + NUMBER + r"\s*(?:/|per)\s*acre\b",
        question,
        re.I,
    )
    match = after or before
    return match.group(1).replace(",", "") if match else None


def choose_tool(state: AgentState) -> dict:
    """Select a tool; use deterministic handling for explicit unit-based cases."""
    question = state["question"]
    q = question.lower()

    # A cost in PKR/acre is NOT a cost in PKR/bag.
    # If either part is missing, the calculator asks for missing information.
    area_rate = find_pkr_per_acre(q)
    if "fertiliz" in q and area_rate is not None:
        return {
            "selected_tool": "calculator",
            "calculation": "fertilizer_area_cost",
            "arg1": find_area_acres(q),
            "arg2": area_rate,
        }

    # Recognize fertilizer acreage cost requests even if the price is missing.
    # This avoids a Gemini call for a straightforward missing-input case.
    area = find_area_acres(q)
    cost_intent = any(word in q for word in ("cost", "price", "expense"))
    if "fertiliz" in q and cost_intent and area is not None:
        return {
            "selected_tool": "calculator",
            "calculation": "fertilizer_area_cost",
            "arg1": area,
            "arg2": None,
        }

    # Existing crop production shortcut: area x kg/acre.
    rate_match = re.search(
        NUMBER + r"\s*(?:kg|kilograms?)\s*(?:/|per)\s*acre\b", q
    )
    area_match = re.search(NUMBER + r"\s*acres?\b", q)
    production_intent = any(
        word in q for word in ("production", "total yield", "harvest")
    )
    if rate_match and area_match and production_intent:
        return {
            "selected_tool": "calculator",
            "calculation": "yield_production",
            "arg1": area_match.group(1).replace(",", ""),
            "arg2": rate_match.group(1).replace(",", ""),
        }

    if not API_KEY:
        return {
            "selected_tool": "unsupported",
            "answer": "Gemini API key is missing.",
        }

    instruction = """You are the tool router for AgriAssist-RX.

Choose exactly one tool and one calculation operation.

RAG: Agricultural knowledge, wheat diseases, crop explanations, and effects
of weather on crop diseases.

WEATHER: Current weather for Pakistani cities. Historical weather and
long-range forecasts are not supported.

CALCULATOR: Only use explicitly supplied numeric inputs with compatible units.

fertilizer_quantity: arg1 = acres; arg2 = bags per acre.
fertilizer_cost: arg1 = number of bags; arg2 = PKR per bag.
fertilizer_area_cost: arg1 = acres; arg2 = PKR per acre.
seed_requirement: arg1 = acres; arg2 = kg of seed per acre.
area_conversion: arg1 = acres; arg2 = null.
irrigation_volume: arg1 = acres; arg2 = water depth in millimetres.
yield_production: arg1 = acres; arg2 = production rate in kg per acre.

Important: Never treat acres as bags or PKR/acre as PKR/bag.
Select fertilizer_area_cost when the question asks for fertilizer costs
using an acreage and a price per acre. Select fertilizer_cost ONLY when
prices and amounts are expressed per bag and in bags.

If a required value is missing or uses an incompatible unit, select
CALCULATOR with the appropriate operation and set the missing argument to null.
Do not invent values, change units, or calculate the answer yourself.

UNSUPPORTED: Live market prices, live news, missing images, external policy
announcements, legal pesticide dosage recommendations, or requests needing
unavailable external information. If a required market price was not provided,
choose UNSUPPORTED.

For non-calculator tools use calculation = none, arg1 = null, arg2 = null.
Return structured data only.

User question:
""" + question

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
            decision = AgentDecision.model_validate(response.parsed)
        else:
            decision = AgentDecision.model_validate_json(response.text)
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
                "The AI router is temporarily unavailable. Try again later. "
                f"Technical details: {error}"
            ),
        }
    finally:
        if client is not None:
            client.close()


def numbers_in_question(question: str) -> set[Decimal]:
    """Collect numeric values explicitly present in the question."""
    pattern = r"(?<![A-Za-z0-9_.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z0-9_.])"
    values = set()
    for text in re.findall(pattern, question):
        try:
            number = Decimal(text.replace(",", ""))
            if number.is_finite():
                values.add(number)
        except InvalidOperation:
            pass
    return values


def verified_argument(value, question: str) -> str | None:
    """Reject inputs invented by the language model."""
    if value is None:
        return None
    try:
        number = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not number.is_finite() or number not in numbers_in_question(question):
        return None
    return str(number)


def calculator_node(state: AgentState) -> dict:
    question = state["question"]
    operation = state.get("calculation", "none")
    arg1 = verified_argument(state.get("arg1"), question)
    arg2 = verified_argument(state.get("arg2"), question)

    if arg1 is None or (operation != "area_conversion" and arg2 is None):
        return {
            "answer": (
                "I need the required numerical inputs and their units to perform "
                "this calculation. Please provide them explicitly."
            ),
            "sources": [],
            "supported": False,
        }

    try:
        if operation == "fertilizer_quantity":
            result = fertilizer_quantity(arg1, arg2)
            answer = (
                f"Fertilizer required: {result['total_bags']} bags.\n"
                f"Inputs: {arg1} acres at {arg2} bags per acre."
            )

        elif operation == "fertilizer_area_cost":
            area = Decimal(arg1)
            rate = Decimal(arg2)
            if area <= 0:
                raise ValueError("Area must be greater than zero.")
            if rate < 0:
                raise ValueError("Cost per acre cannot be negative.")
            total = area * rate
            answer = (
                f"Total fertilizer cost: PKR {total:,.2f}\n"
                f"Inputs: {area} acres at PKR {rate:,.2f} per acre."
            )

        elif operation == "fertilizer_cost":
            # Only number-of-bags x price-per-bag, not acreage pricing.
            if re.search(r"\b(?:pkr|rs\.?)\s*(?:/|per)\s*acre\b", question, re.I):
                raise ValueError("A price per acre cannot be used as a price per bag.")
            result = fertilizer_cost(arg1, arg2)
            answer = (
                f"Total fertilizer cost: PKR {result['total_cost_pkr']:,.2f}\n"
                f"Inputs: {arg1} bags at PKR {arg2} per bag."
            )

        elif operation == "seed_requirement":
            result = seed_requirement(arg1, arg2)
            answer = (
                f"Total seed required: {result['total_seed_kg']} kg.\n"
                f"Inputs: {arg1} acres at {arg2} kg per acre."
            )

        elif operation == "area_conversion":
            result = acres_to_hectares(arg1)
            answer = f"{arg1} acres = {result['hectares']:.4f} hectares."

        elif operation == "irrigation_volume":
            result = irrigation_volume(arg1, arg2)
            answer = (
                f"Theoretical irrigation volume: {result['volume_m3']:.2f} m³.\n"
                f"Inputs: {arg1} acres with {arg2} mm applied depth.\n"
                "Excludes irrigation losses."
            )

        elif operation == "yield_production":
            area = Decimal(arg1)
            rate = Decimal(arg2)
            if area <= 0:
                raise ValueError("Area must be positive.")
            if rate < 0:
                raise ValueError("Yield rate cannot be negative.")
            total_kg = area * rate
            answer = (
                f"Total crop production: {total_kg:,.2f} kg.\n"
                f"Inputs: {arg1} acres at {arg2} kg per acre."
            )

        else:
            return {
                "answer": "This calculation is not supported yet.",
                "sources": [],
                "supported": False,
            }

        return {"answer": answer, "sources": [], "supported": True}

    except (ValueError, ArithmeticError) as error:
        return {
            "answer": f"Calculation error: {error}",
            "sources": [],
            "supported": False,
        }


def unsupported_node(state: AgentState) -> dict:
    return {
        "answer": state.get("answer") or (
            "I cannot fulfill this request with the currently available tools."
        ),
        "sources": [],
        "supported": False,
    }


# The standalone four-route workflow remains available.
workflow = StateGraph(AgentState)
workflow.add_node("router", choose_tool)
workflow.add_node("rag", rag_tool)
workflow.add_node("weather", weather_tool)
workflow.add_node("calculator", calculator_node)
workflow.add_node("unsupported", unsupported_node)
workflow.add_edge(START, "router")
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
for node in ("rag", "weather", "calculator", "unsupported"):
    workflow.add_edge(node, END)
app = workflow.compile()


def main() -> None:
    print("=" * 55)
    print("AGRIASSIST-RX: FOUR-TOOL LANGGRAPH AGENT")
    print("=" * 55)
    print("\nType 'exit' to stop.\n")

    try:
        while True:
            question = input("Your question: ").strip()
            if question.lower() in ("exit", "quit"):
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
                print("Selected tool:", result["selected_tool"])
                print("-" * 55)
                print("\nAGRIASSIST-RX ANSWER:")
                print(result["answer"])
                if result.get("sources"):
                    print("\nEVIDENCE SOURCES:")
                    for source in result["sources"]:
                        print(
                            f"[{source['id']}] {source['source']} "
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
