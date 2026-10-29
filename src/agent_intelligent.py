
"""
AgriAssist-RX
Phase 4, Step 6: Gemini-Based Intelligent Router

Features:
1. Gemini classifies user intent.
2. LangGraph routes to the correct tool.
3. Agricultural RAG uses our existing chatbot.
4. Weather uses the existing Open-Meteo tool.
5. Unsupported questions receive a safe response.
6. API failures do not trigger an unreliable guess.

Existing project files remain unchanged.
"""

import os
import re

from enum import Enum
from pathlib import Path
from typing import TypedDict

import requests

from dotenv import load_dotenv
from google import genai
from google.genai import types

from langgraph.graph import StateGraph, START, END


# ==========================================
# STEP 1: Project configuration
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.1-flash-lite"


# ==========================================
# STEP 2: Define the allowed tool choices
# ==========================================

class ToolChoice(str, Enum):
    RAG = "rag"
    WEATHER = "weather"
    UNSUPPORTED = "unsupported"


# ==========================================
# STEP 3: Shared LangGraph state
# ==========================================

class AgentState(TypedDict):
    question: str
    selected_tool: str
    answer: str
    sources: list
    supported: bool


# ==========================================
# STEP 4: Gemini intelligent router
# ==========================================

def choose_tool(state: AgentState):

    if not API_KEY:
        return {
            "selected_tool": "unsupported",
            "answer": (
                "Gemini API key is missing. "
                "Check your private .env file."
            )
        }

    question = state["question"]

    instruction = """
You are the intent router for AgriAssist-RX,
an agricultural assistant.

Select exactly one available tool.

RAG:
Choose this for agricultural knowledge questions
such as wheat rust, crop diseases, symptoms,
disease spread, disease prevention, and general
agricultural explanations.

A question about how weather influences crops
is an agricultural knowledge question, so use RAG.

WEATHER:
Choose this only when the user asks for actual
current weather measurements, such as temperature,
humidity, rainfall, precipitation or wind, for
a location in Pakistan.

A missing city can still use WEATHER so that
the weather tool can ask the user for a city.

UNSUPPORTED:
Choose this if neither tool can fulfill the request.
Examples include current market prices, live news,
unprovided image analysis, unrelated trivia,
or requests requiring capabilities we do not have.

If a request requires multiple different tools
or a historical forecast that is not available,
choose UNSUPPORTED for this prototype.

Do not answer the question.
Only select a tool.

User question:
""" + question

    client = None

    try:

        client = genai.Client(api_key=API_KEY)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=instruction,
            config=types.GenerateContentConfig(
                response_mime_type="text/x.enum",
                response_schema=ToolChoice,
                temperature=0,
            ),
        )

        if not response.text:
            raise ValueError("Empty router response.")

        selected = response.text.strip().lower()

        allowed = {
            "rag",
            "weather",
            "unsupported",
        }

        if selected not in allowed:
            raise ValueError(
                f"Unexpected tool choice: {selected}"
            )

        return {
            "selected_tool": selected
        }

    except Exception as error:

        # Never guess a tool after an API failure.
        return {
            "selected_tool": "unsupported",
            "answer": (
                "The intelligent router is temporarily "
                "unavailable. Please try again later. "
                f"Technical details: {error}"
            )
        }

    finally:

        if client is not None:
            client.close()


# ==========================================
# STEP 5: Real agricultural RAG tool
# ==========================================

def rag_tool(state: AgentState):

    from rag import generate_answer

    result = generate_answer(state["question"])

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "supported": result["supported"],
    }


# ==========================================
# STEP 6: Extract a Pakistani city name
# ==========================================

def extract_city(question):

    pattern = (
        r"\b(?:in|for|at)\s+"
        r"([A-Za-z][A-Za-z\s-]*?)"
        r"(?:\s+(?:today|tonight|right now|"
        r"now|currently))?"
        r"\s*[?.!]*$"
    )

    match = re.search(
        pattern,
        question.strip(),
        flags=re.IGNORECASE
    )

    if not match:
        return None

    city = match.group(1).strip()

    return city or None


# ==========================================
# STEP 7: Real weather tool
# ==========================================

def weather_tool(state: AgentState):

    from weather_tool import (
        get_current_weather,
        format_weather,
    )

    city = extract_city(state["question"])

    if not city:

        return {
            "answer": (
                "Please specify a Pakistani city. "
                "Example: What is the temperature "
                "in Lahore?"
            ),
            "sources": [],
            "supported": False,
        }

    try:

        weather = get_current_weather(
            city,
            country_code="PK"
        )

        return {
            "answer": format_weather(weather),
            "sources": [],
            "supported": True,
        }

    except (
        requests.RequestException,
        ValueError,
    ) as error:

        return {
            "answer": (
                "Unable to retrieve current weather "
                f"for {city}. Details: {error}"
            ),
            "sources": [],
            "supported": False,
        }


# ==========================================
# STEP 8: Unsupported question handler
# ==========================================

def unsupported_tool(state: AgentState):

    existing_answer = state.get("answer", "")

    if existing_answer:
        answer = existing_answer

    else:
        answer = (
            "This request cannot be handled by "
            "the available AgriAssist-RX tools. "
            "I can answer agricultural questions "
            "using our documents or retrieve "
            "current weather for Pakistani cities."
        )

    return {
        "answer": answer,
        "sources": [],
        "supported": False,
    }


# ==========================================
# STEP 9: Route between the three nodes
# ==========================================

def route_question(state: AgentState):

    return state["selected_tool"]


# ==========================================
# STEP 10: Build the LangGraph workflow
# ==========================================

workflow = StateGraph(AgentState)

workflow.add_node("router", choose_tool)

workflow.add_node("rag", rag_tool)

workflow.add_node("weather", weather_tool)

workflow.add_node("unsupported", unsupported_tool)

workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    route_question,
    {
        "rag": "rag",
        "weather": "weather",
        "unsupported": "unsupported",
    },
)

workflow.add_edge("rag", END)
workflow.add_edge("weather", END)
workflow.add_edge("unsupported", END)

app = workflow.compile()


# ==========================================
# STEP 11: Interactive application
# ==========================================

def main():

    print("=" * 55)
    print("AGRIASSIST-RX: INTELLIGENT LANGGRAPH AGENT")
    print("=" * 55)

    print("\nType 'exit' to stop.\n")

    try:

        while True:

            question = input(
                "Your question: "
            ).strip()

            if question.lower() in ("exit", "quit"):
                print("\nGoodbye!")
                break

            if not question:
                continue

            try:

                result = app.invoke({
                    "question": question,
                    "selected_tool": "",
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
                    "\nUnable to process the question."
                )

                print("Error:", error)

                print("Please try again.\n")

    finally:

        # Close the separate RAG Gemini client
        # if the RAG module was loaded.
        import sys

        if "rag" in sys.modules:
            from rag import client
            client.close()


if __name__ == "__main__":
    main()

