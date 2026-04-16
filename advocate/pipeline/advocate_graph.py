"""
advocate_graph.py  —  LangGraph Orchestration
Five-agent ADVOCATE pipeline with model threaded through state.

Pipeline:
  parse_case → [employer_agent ‖ employee_agent] → irac_evaluator → gap_report
"""

import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from advocate.agents.parser_agent import parse_case
from advocate.agents.employer_agent import build_employer_arguments
from advocate.agents.employee_agent import build_employee_arguments
from advocate.agents.irac_evaluator import evaluate
from advocate.agents.gap_report import generate_gap_report


# ─── State Schema ─────────────────────────────────────────────────────────────

class AdvocateState(TypedDict):
    # Input
    case_brief: str
    model: str                     # LLM model to use for all agents in this run

    # Agent outputs
    parsed_case: dict
    employer_args: dict
    employee_args: dict
    evaluation: dict
    gap_report: dict

    # Metadata
    errors: Annotated[list[str], operator.add]
    steps_completed: Annotated[list[str], operator.add]


# ─── Node Functions ───────────────────────────────────────────────────────────

def node_parse_case(state: AdvocateState) -> dict:
    try:
        parsed = parse_case(state["case_brief"], model=state["model"])
        return {"parsed_case": parsed, "steps_completed": ["parse_case"]}
    except Exception as e:
        return {"parsed_case": {}, "errors": [f"parse_case: {e}"], "steps_completed": ["parse_case"]}


def node_employer_agent(state: AdvocateState) -> dict:
    """ISOLATION: receives only parsed_case and model — zero shared context with employee agent."""
    try:
        args = build_employer_arguments(state["parsed_case"], model=state["model"])
        return {"employer_args": args, "steps_completed": ["employer_agent"]}
    except Exception as e:
        return {
            "employer_args": {"side": "employer", "claims": [], "error": str(e)},
            "errors": [f"employer_agent: {e}"],
            "steps_completed": ["employer_agent"],
        }


def node_employee_agent(state: AdvocateState) -> dict:
    """ISOLATION: receives only parsed_case and model — zero shared context with employer agent."""
    try:
        args = build_employee_arguments(state["parsed_case"], model=state["model"])
        return {"employee_args": args, "steps_completed": ["employee_agent"]}
    except Exception as e:
        return {
            "employee_args": {"side": "employee", "claims": [], "error": str(e)},
            "errors": [f"employee_agent: {e}"],
            "steps_completed": ["employee_agent"],
        }


def node_irac_evaluator(state: AdvocateState) -> dict:
    try:
        eval_result = evaluate(
            case=state["parsed_case"],
            employer_args=state["employer_args"],
            employee_args=state["employee_args"],
            model=state["model"],
        )
        return {"evaluation": eval_result, "steps_completed": ["irac_evaluator"]}
    except Exception as e:
        return {"evaluation": {}, "errors": [f"irac_evaluator: {e}"], "steps_completed": ["irac_evaluator"]}


def node_gap_report(state: AdvocateState) -> dict:
    try:
        report = generate_gap_report(
            case=state["parsed_case"],
            employer_args=state["employer_args"],
            employee_args=state["employee_args"],
            evaluation=state["evaluation"],
            model=state["model"],
        )
        return {"gap_report": report, "steps_completed": ["gap_report"]}
    except Exception as e:
        return {"gap_report": {"error": str(e)}, "errors": [f"gap_report: {e}"], "steps_completed": ["gap_report"]}


# ─── Graph Construction ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AdvocateState)
    graph.add_node("parse_case", node_parse_case)
    graph.add_node("employer_agent", node_employer_agent)
    graph.add_node("employee_agent", node_employee_agent)
    graph.add_node("irac_evaluator", node_irac_evaluator)
    graph.add_node("gap_report", node_gap_report)

    graph.set_entry_point("parse_case")
    graph.add_edge("parse_case", "employer_agent")
    graph.add_edge("parse_case", "employee_agent")
    graph.add_edge("employer_agent", "irac_evaluator")
    graph.add_edge("employee_agent", "irac_evaluator")
    graph.add_edge("irac_evaluator", "gap_report")
    graph.add_edge("gap_report", END)
    return graph.compile()


# ─── Public API ───────────────────────────────────────────────────────────────

def run_pipeline(case_brief: str, model: str | None = None) -> AdvocateState:
    """
    Run the full ADVOCATE pipeline on a case brief.

    Args:
        case_brief: Raw text of the case brief.
        model:      LLM model ID for all agents. Defaults to ADVOCATE_MODEL env var.
    """
    import os
    model = model or os.getenv("ADVOCATE_MODEL", "gpt-4o")
    graph = build_graph()
    initial: AdvocateState = {
        "case_brief": case_brief,
        "model": model,
        "parsed_case": {},
        "employer_args": {},
        "employee_args": {},
        "evaluation": {},
        "gap_report": {},
        "errors": [],
        "steps_completed": [],
    }
    return graph.invoke(initial)
