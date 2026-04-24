"""LangGraph orchestration for the five-agent ADVOCATE pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

from advocate.agents.employee_agent import build_employee_arguments
from advocate.agents.employer_agent import build_employer_arguments
from advocate.agents.gap_report import generate_gap_report
from advocate.agents.irac_evaluator import evaluate
from advocate.agents.parser_agent import parse_case
from advocate.settings import get_default_model


class AdvocateState(TypedDict):
    case_brief: str
    model: str
    parsed_case: dict
    employer_args: dict
    employee_args: dict
    evaluation: dict
    gap_report: dict
    errors: Annotated[list[str], operator.add]
    steps_completed: Annotated[list[str], operator.add]


def node_parse_case(state: AdvocateState) -> dict:
    try:
        parsed = parse_case(state["case_brief"], model=state["model"])
        return {"parsed_case": parsed, "steps_completed": ["parse_case"]}
    except Exception as exc:
        return {
            "parsed_case": {},
            "errors": [f"parse_case: {exc}"],
            "steps_completed": ["parse_case"],
        }


def node_employer_agent(state: AdvocateState) -> dict:
    try:
        arguments = build_employer_arguments(state["parsed_case"], model=state["model"])
        return {"employer_args": arguments, "steps_completed": ["employer_agent"]}
    except Exception as exc:
        return {
            "employer_args": {"side": "employer", "claims": [], "error": str(exc)},
            "errors": [f"employer_agent: {exc}"],
            "steps_completed": ["employer_agent"],
        }


def node_employee_agent(state: AdvocateState) -> dict:
    try:
        arguments = build_employee_arguments(state["parsed_case"], model=state["model"])
        return {"employee_args": arguments, "steps_completed": ["employee_agent"]}
    except Exception as exc:
        return {
            "employee_args": {"side": "employee", "claims": [], "error": str(exc)},
            "errors": [f"employee_agent: {exc}"],
            "steps_completed": ["employee_agent"],
        }


def node_irac_evaluator(state: AdvocateState) -> dict:
    try:
        evaluation = evaluate(
            case=state["parsed_case"],
            employer_args=state["employer_args"],
            employee_args=state["employee_args"],
            model=state["model"],
        )
        return {"evaluation": evaluation, "steps_completed": ["irac_evaluator"]}
    except Exception as exc:
        return {
            "evaluation": {},
            "errors": [f"irac_evaluator: {exc}"],
            "steps_completed": ["irac_evaluator"],
        }


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
    except Exception as exc:
        return {
            "gap_report": {"error": str(exc)},
            "errors": [f"gap_report: {exc}"],
            "steps_completed": ["gap_report"],
        }


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


def run_pipeline(case_brief: str, model: str | None = None) -> AdvocateState:
    """Run the full ADVOCATE pipeline on a case brief."""
    graph = build_graph()
    selected_model = model or get_default_model()
    initial_state: AdvocateState = {
        "case_brief": case_brief,
        "model": selected_model,
        "parsed_case": {},
        "employer_args": {},
        "employee_args": {},
        "evaluation": {},
        "gap_report": {},
        "errors": [],
        "steps_completed": [],
    }
    return graph.invoke(initial_state)
