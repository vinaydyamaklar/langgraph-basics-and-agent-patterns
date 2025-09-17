import os
import operator

from pprint import pprint

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from typing import Annotated, List
from typing_extensions import Literal, TypedDict

from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

llm=ChatOpenAI(model="gpt-4o")


class Section(BaseModel):
    name: str = Field(description="Name for this section of the report")
    description: str = Field(description="Brief overview of main topics and conepts of the section")


class Sections(BaseModel):
    section: List[Section] = Field(description="Secrions of the report")


# Augment the LLM with scheama for structured output
planner=llm.with_structured_output(Sections)


### creating workers Dynamically in LangGraph
# Because Orchestrator-worker workflows are very common, Send API is provided by LangGraph to help.
# from langgraph.constants import Send
# it helps us create worke nodes on the fly.
# Even though all workers have their own states they will be writing to a common state(Shared state)

# Graph state
class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_section: Annotated[
        list, operator.add
    ] # all workers write to this in parallel
    final_report: str # Final report


class WorkerState(TypedDict):
    section: Section
    completed_section: Annotated[list, operator.add]


def orchestrator(state: State):
    """Orchestrator that generates plan for the report"""

    # Generate queries
    report_sections = planner.invoke(
        [
            SystemMessage(content="Generate a plan for the report."),
            HumanMessage(content=f"Here is the report topic: {state['topic']}")
        ]
    )

    print("Report Sections:", report_sections)

    return {"sections": report_sections.section}


def llm_call(state: WorkerState):
    """Worker writes a section of the report"""

    # Generate section
    section = llm.invoke(
        [
            SystemMessage(
                content=f"Write a section following the provided name and description. Include no premable for each section. Use markdown formating"
            ),
            HumanMessage(
                content=f"Here is the secion name: {state['section']} and description: {state["section"].description}"
            )
        ]
    )

    return {"completed_section": [section.content]}


def assign_workers(state: State):
    """Assign worker to each section in plan"""

    # Kick off section writing in parallel via send() API
    return [Send("llm_call", {"section": s}) for s in state['sections']]


def synthesizer(state: State):
    """Synthesize full report from sections"""

    #  List of completed sections
    completed_sections = state["completed_section"]

    # Format completed section to str to use as context for final sections
    completed_report_sections = "\n\n--\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}


# building graph
orechestrator_worker_builder = StateGraph(State)

# Add nodes
orechestrator_worker_builder.add_node("orchestrator", orchestrator)
orechestrator_worker_builder.add_node("llm_call", llm_call)
orechestrator_worker_builder.add_node("synthesizer", synthesizer)

# Add edges
orechestrator_worker_builder.add_edge(START, "orchestrator")
orechestrator_worker_builder.add_conditional_edges(
    "orchestrator", assign_workers, ["llm_call"]
)
orechestrator_worker_builder.add_edge("llm_call", "synthesizer")
orechestrator_worker_builder.add_edge("synthesizer", END)


orechestrator_worker = orechestrator_worker_builder.compile()


if __name__ == "__main__":
    state = orechestrator_worker.invoke({"topic": "Create a report on 'Effect of No Code tools on software engineering!'"})

    print(state['final_report'])