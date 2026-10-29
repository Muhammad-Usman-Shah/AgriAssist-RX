
"""
AgriAssist-RX
Phase 4: First LangGraph Workflow

Demonstrates:
1. Shared graph state
2. Processing nodes
3. Conditional routing
4. Graph execution

This is a local demonstration.
No Gemini API requests are made.
"""

from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    START,
    END,
)


# STEP 1: Define shared state

class AgentState(TypedDict):
    question: str
    selected_tool: str
    answer: str


# STEP 2: Create a routing node

def choose_tool(state: AgentState):

    question = state["question"].lower()

    weather_words = [
        "weather",
        "temperature",
        "rain",
        "forecast",
    ]

    if any(
        word in question
        for word in weather_words
    ):
        selected = "weather"
    else:
        selected = "rag"

    return {
        "selected_tool": selected
    }


# STEP 3: Create demonstration tools

def rag_tool(state: AgentState):

    return {
        "answer": (
            "RAG tool selected. "
            "Agricultural document retrieval "
            "will be connected here."
        )
    }


def weather_tool(state: AgentState):

    return {
        "answer": (
            "Weather tool selected. "
            "A live weather API will be "
            "connected here."
        )
    }


# STEP 4: Define conditional routing

def route_question(state: AgentState):

    return state["selected_tool"]


# STEP 5: Build the graph

workflow = StateGraph(AgentState)

workflow.add_node("choose_tool", choose_tool)
workflow.add_node("rag", rag_tool)
workflow.add_node("weather", weather_tool)

workflow.add_edge(START, "choose_tool")

workflow.add_conditional_edges(
    "choose_tool",
    route_question,
    {
        "rag": "rag",
        "weather": "weather",
    },
)

workflow.add_edge("rag", END)
workflow.add_edge("weather", END)

app = workflow.compile()


# STEP 6: Test the workflow

def main():

    print("=" * 55)
    print("AGRIASSIST-RX: FIRST LANGGRAPH WORKFLOW")
    print("=" * 55)

    print("\nType 'exit' to stop.\n")

    while True:

        question = input("Your question: ").strip()

        if question.lower() in ("exit", "quit"):
            break

        if not question:
            continue

        result = app.invoke({
            "question": question,
            "selected_tool": "",
            "answer": "",
        })

        print("\nSelected tool:", result["selected_tool"])
        print("Response:", result["answer"])
        print("-" * 55)


if __name__ == "__main__":
    main()
