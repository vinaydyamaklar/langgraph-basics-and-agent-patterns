import os
from pprint import PrettyPrinter
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from typing_extensions import TypedDict

from IPython.display import Image, display

from langgraph.graph import StateGraph, START, END


load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="groq/compound-mini")

# Graph state
class State(TypedDict):
    topic: str
    characters: str
    settings: str
    premises: str
    story_intro: str


def generate_character(state: State):
    """Generate Character description"""
    msg = llm.invoke(f"Create two character names and brief traits for a story about {state["topic"]}")
    return {"characters": msg.content}


def generate_settings(state: State):
    """Generate a story settings"""
    msg = llm.invoke(F"Describe a vivid setting for a story about {state['topic']}")
    return {"settings": msg.content}


def generate_premise(state: State):
    """Generate a story premise"""
    msg = llm.invoke(f"Write a one-sentence plot premise for a story about {state['topic']}")
    return {"premises": msg.content}


def combine_element(state: State):
    """Combine characters, settings, and premise into an intro"""
    msg = llm.invoke(
        f"Write a short story introduction using these elements: \n"
        f"Characters: {state["characters"]}\n"
        f"Settings: {state["settings"]}"
        f"Premise: {state["premises"]}"
    )

    return {"story_intro": msg.content}


# build the graph
graph = StateGraph(State)

# add nodes
graph.add_node("character", generate_character)
graph.add_node("settings", generate_settings)
graph.add_node("premise", generate_premise)
graph.add_node("combine", combine_element)


# add edges
graph.add_edge(START, "character")
graph.add_edge(START, "settings")
graph.add_edge(START, "premise")
graph.add_edge("character", "combine")
graph.add_edge("settings", "combine")
graph.add_edge("premise", "combine")
graph.add_edge("combine", END)

# Compiled graph
compiled_graph = graph.compile()

state = {"topic": "time travel"}
result = compiled_graph.invoke(state)
PrettyPrinter(indent=4).pprint(result)