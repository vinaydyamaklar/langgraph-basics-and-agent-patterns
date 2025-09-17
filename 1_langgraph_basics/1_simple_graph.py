import random

from typing import Literal # Literal basically meany costants

from typing_extensions import TypedDict # TypedDict helps to represents the value in the form of dictionary by inherited class

from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END



class State(TypedDict):
    graph_info: str

def start_play(state: State):
    print("Start Play node has been called")
    return {"graph_info": state['graph_info'] + "I am planning to play"}


def cricket(state: State):
    print("Cricket node has been called")
    return {"graph_info": state["graph_info"] + "I am playing cricket"}


def badminton(state: State):
    print("Badminton node has been called")
    return {"graph_info": state["graph_info"] + "I am playing badminton"}


def random_play(state: State) -> Literal['cricket', 'badminton']:
    graph_info = state["graph_info"]

    if random.random() > 0.5:
        return "cricket"
    else:
        return "badminton"


## building the graph
graph = StateGraph(State)

## adding the nodes
graph.add_node("start_play", start_play)
graph.add_node("cricket", cricket)
graph.add_node("badminton", badminton)

## Schdule the flow of the graph
graph.add_edge(START, "start_play")
graph.add_conditional_edges("start_play", random_play)
graph.add_edge("cricket", END)
graph.add_edge("badminton", END)

## compile the graph
graph_builder = graph.compile()

## view
display(Image(graph_builder.get_graph().draw_mermaid_png()))

## graph invokation

graph_builder.invoke({"graph_info": "Hi My name is vinay"})