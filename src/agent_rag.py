
"""
AgriAssist-RX
Phase 4: LangGraph Agent with RAG and Weather

Features:
1. Keyword-based tool routing
2. Real agricultural RAG using Gemini
3. Live weather information using Open-Meteo
4. PDF evidence citations
5. Basic location and API error handling

Weather lookup currently supports Pakistani cities.
"""

import re
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ==========================================
# STEP 1: Shared agent state
# ==========================================

class AgentState(TypedDict):
    question: str
    selected_tool: str
    answer: str
    sources: list
    supported: bool


# ==========================================
# STEP 2: Choose the appropriate tool
# ==========================================

def choose_tool(state: AgentState):

    question = state["question"].lower()

    weather_keywords = [
        "weather",
        "temperature",
        "forecast",
        "rainfall",
        "humidity",
        "wind speed",
        "precipitation",
        "degrees celsius",
    ]

    if any(word in question for word in weather_keywords):
        selected = "weather"
    else:
        selected = "rag"

    return {"selected_tool": selected}


# ==========================================
# STEP 3: Real agricultural RAG tool
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
# STEP 4: Extract city from question
# ==========================================

def extract_city(question):

    """
    Examples:
    'What is the weather in Lahore?'
       -> Lahore

    'Temperature in Islamabad today?'
       -> Islamabad

    'Weather for Quetta'
       -> Quetta
    """

    pattern = (
        r"\b(?:in|for|at)\s+"
        r"([A-Za-z][A-Za-z\s-]*?)"
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

    # Remove trailing time expressions.
    city = re.sub(
        r"\s+(?:today|tomorrow|tonight|now|"
        r"right now|currently)$",
        "",
        city,
        flags=re.IGNORECASE
    ).strip()

    return city if city else None


# ==========================================
# STEP 5: Real weather tool
# ==========================================

def weather_tool(state: AgentState):

    import requests

    from weather_tool import (
        get_current_weather,
        format_weather
    )

    question = state["question"]

    city = extract_city(question)

    if not city:

        return {
            "answer": (
                "Please include a Pakistani city name. "
                "For example: "
                "'What is the temperature in Lahore?'"
            ),
            "sources": [],
            "supported": False,
        }

    try:

        weather = get_current_weather(
            city,
            country_code="PK"
        )

        answer = format_weather(weather)

        return {
            "answer": answer,
            "sources": [],
            "supported": True,
        }

    except (
        requests.RequestException,
        ValueError
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
# STEP 6: Graph routing
# ==========================================

def route_question(state: AgentState):

    return state["selected_tool"]


# ==========================================
# STEP 7: Build LangGraph workflow
# ==========================================

workflow = StateGraph(AgentState)

workflow.add_node("choose_tool", choose_tool)

workflow.add_node("rag", rag_tool)

workflow.add_node("weather", weather_tool)

workflow.add_edge(
    START,
    "choose_tool"
)

workflow.add_conditional_edges(
    "choose_tool",
    route_question,
    {
        "rag": "rag",
        "weather": "weather",
    }
)

workflow.add_edge("rag", END)

workflow.add_edge("weather", END)

app = workflow.compile()


# ==========================================
# STEP 8: Interactive application
# ==========================================

def main():

    print("=" * 55)
    print("AGRIASSIST-RX: RAG + LIVE WEATHER AGENT")
    print("=" * 55)

    print("\nType 'exit' to stop.\n")

    try:

        while True:

            question = input(
                "Your question: "
            ).strip()

            if question.lower() in (
                "exit", "quit"
            ):
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
                    "\nUnable to process this question."
                )

                print("Error:", error)

                print("Please try again.\n")

    finally:

        # Close the Gemini client if RAG was used.
        import sys

        if "rag" in sys.modules:

            from rag import client

            client.close()


if __name__ == "__main__":
    main()
