import os
from pprint import PrettyPrinter
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from typing_extensions import Literal, TypedDict

from pydantic import BaseModel, Field

from IPython.display import Image, display

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="groq/compound-mini")


# Schema for structured output to use as routing logic
class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(description="The next step in the routing process")

## Augment the llm with schema for structured output
router = llm.with_structured_output(Route)

## state
class State(TypedDict):
    input: str
    decision: str
    output: str


# Nodes
def llm_call_1(state: State):
    """Write a story"""

    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_2(state: State):
    """Write a Joke"""

    result = llm.invoke(state["input"])
    return {"output": result.content}

def llm_call_3(state: State):
    """Write a poem"""

    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_router(state: State):
    """Route the input to appropriate node"""

    decision = router.invoke([
        SystemMessage(
            content="Route the input to story, joke or poem based on the user's request"
        ),
        HumanMessage(
            content=state['input']
        )
    ])
    return {"desision": decision.step}


def route_decision(state: State):
    # returns the node name you want to visit next
    if state['decision'] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "poem":
        return "llm_call_3"


## build the workflow
router_builder = StateGraph(State)

## Add node
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)


## Add edges to connect nodes
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3"
    }
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

router_workflow = router_builder.compile()

state = router_workflow.invoke({"input": "Write me a joke about agentic AI system!"})