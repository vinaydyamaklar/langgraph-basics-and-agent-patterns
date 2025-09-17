"""
Prompt Chaining is a technique in natural language processing where multiple prompts 
are seequnced together to guide model through a complex task or reasoning process.
Instead of relying on single prompt to achieve a desired outcome, prompt chaining 
breaks the task into smaller manageble steps, with each step building on the previous
one.
This approach can improve accuracy, coherence, and control when working with 
large language models.

LangGraph, is framework designed to facilitate structured interactions with language 
models, making it an excellent tool for implementing prompt chaining.
It allows you to define a graph of nodes (representing indivdual prompts or tasks) and 
edges (representing the flow of information between them).
This structure enables dynamic, multi-step conversations or workflows, where the 
output of one node can feed into the input of the next.
"""


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
    story: str
    improved_story: str
    final_story: str


# Nodes
def generate_story(state: State):
    msg=llm.invoke(f"Write a one sentence story premise about {state["topic"]}")
    return {"story": msg.content}

def check_conflict(state: State):
    if "?" in state["story"] or "!" in state["story"]:
        return "Fail"
    return "Pass"

def improve_story(state: State):
    msg=llm.invoke(f"Enhance this stry premise with vivid details: {state["story"]}")
    return {"improved_story": msg.content}

def polish_story(state: State):
    msg=llm.invoke(f"Add an unexpected twist to this story premise: {state["improved_story"]}")
    return {"final_story": msg.content}


# Build the graph
graph = StateGraph(State)

graph.add_node("generate", generate_story)
graph.add_node("improve", improve_story)
graph.add_node("polish", polish_story)


# Define the edges
graph.add_edge(START, "generate")
graph.add_conditional_edges("generate", check_conflict, {"Pass": "improve", "Fail": "generate"})
graph.add_edge("improve", "polish")
graph.add_edge("polish", END)


# compile the graph
compiled_graph = graph.compile()

# visualize the graph (for Jupyter notebook)
# graph_image = compiled_graph.get_graph().draw_mermaid_png()
# display(Image(graph_image))


## Run the graph
state={"topic": "Agentic AI systems"}
result = compiled_graph.invoke(state)
PrettyPrinter(indent=4).pprint(result)