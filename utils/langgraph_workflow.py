from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from utils.llm_followup import answer_followup_with_llm, update_parsed_request_with_llm
from utils.llm_itinerary import generate_itinerary_with_llm
from utils.llm_parser import parse_travel_request_with_llm
from utils.llm_planner import generate_plan
from utils.llm_query_rewriter import rewrite_followup_query, rewrite_initial_query
from utils.rag_retriever import retrieve_travel_knowledge
from utils.travel_graph import run_travel_agent_tools
from utils.user_memory import apply_memory_to_request


StageRunner = Callable[[str, Callable[[], Any]], Any]


class TripGenerationState(TypedDict, total=False):
    user_message: str
    is_update: bool
    current_parsed_request: Optional[Dict[str, Any]]
    current_itinerary: Optional[Dict[str, Any]]
    previous_tool_results: Optional[Dict[str, Any]]
    conversation_history: List[Dict[str, str]]
    stage_runner: Optional[StageRunner]
    rewritten_message: str
    rewrite_record_stage: str
    parsed_request: Dict[str, Any]
    execution_plan: Dict[str, Any]
    tool_results: Dict[str, Any]
    rag_context: Dict[str, Any]
    final_itinerary: Dict[str, Any]


class FollowupQaState(TypedDict, total=False):
    user_message: str
    parsed_request: Dict[str, Any]
    execution_plan: Dict[str, Any]
    tool_results: Dict[str, Any]
    final_itinerary: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    stage_runner: Optional[StageRunner]
    rewritten_message: str
    answer: str


def _run_stage(state: Dict[str, Any], stage: str, fn: Callable[[], Any]) -> Any:
    runner = state.get("stage_runner")
    return runner(stage, fn) if runner else fn()


def _route_generation_mode(state: TripGenerationState) -> str:
    if state.get("is_update") and state.get("current_parsed_request") is not None:
        return "update"
    return "initial"


def _noop(state: TripGenerationState) -> Dict[str, Any]:
    return {}


def _rewrite_initial(state: TripGenerationState) -> Dict[str, Any]:
    rewritten = _run_stage(
        state,
        "initial_rewrite",
        lambda: rewrite_initial_query(state["user_message"]),
    )
    return {
        "rewritten_message": rewritten,
        "rewrite_record_stage": "initial_parse",
    }


def _parse_initial(state: TripGenerationState) -> Dict[str, Any]:
    parsed_request = _run_stage(
        state,
        "parse_request",
        lambda: parse_travel_request_with_llm(
            state["rewritten_message"],
            original_user_input=state["user_message"],
        ),
    )
    parsed_request = apply_memory_to_request(parsed_request, user_message=state["user_message"])
    return {"parsed_request": parsed_request}


def _rewrite_update(state: TripGenerationState) -> Dict[str, Any]:
    rewritten = _run_stage(
        state,
        "followup_rewrite",
        lambda: rewrite_followup_query(
            user_message=state["user_message"],
            parsed_request=state.get("current_parsed_request"),
            current_itinerary=state.get("current_itinerary"),
            conversation_history=state.get("conversation_history", []),
        ),
    )
    return {
        "rewritten_message": rewritten,
        "rewrite_record_stage": "followup_update",
    }


def _update_request(state: TripGenerationState) -> Dict[str, Any]:
    parsed_request = _run_stage(
        state,
        "update_request",
        lambda: update_parsed_request_with_llm(
            current_request=state.get("current_parsed_request") or {},
            user_message=state["user_message"],
            conversation_history=state.get("conversation_history", []),
            current_itinerary=state.get("current_itinerary"),
            rewritten_user_message=state["rewritten_message"],
        ),
    )
    parsed_request = apply_memory_to_request(parsed_request, user_message=state["user_message"])
    return {"parsed_request": parsed_request}


def _plan_trip(state: TripGenerationState) -> Dict[str, Any]:
    execution_plan = _run_stage(
        state,
        "deterministic_plan",
        lambda: generate_plan(state["parsed_request"]),
    )
    return {"execution_plan": execution_plan}


def _run_tools(state: TripGenerationState) -> Dict[str, Any]:
    tool_results = _run_stage(
        state,
        "tools",
        lambda: run_travel_agent_tools(
            state["parsed_request"],
            previous_request=state.get("current_parsed_request"),
            previous_tool_results=state.get("previous_tool_results"),
        ),
    )
    return {"tool_results": tool_results}


def _retrieve_knowledge(state: TripGenerationState) -> Dict[str, Any]:
    rag_context = _run_stage(
        state,
        "rag_retrieval",
        lambda: retrieve_travel_knowledge(
            state["parsed_request"],
            user_message=state.get("rewritten_message", state["user_message"]),
        ),
    )
    return {"rag_context": rag_context}


def _generate_itinerary(state: TripGenerationState) -> Dict[str, Any]:
    itinerary = _run_stage(
        state,
        "itinerary_llm",
        lambda: generate_itinerary_with_llm(
            parsed_request=state["parsed_request"],
            execution_plan=state["execution_plan"],
            tool_results=state["tool_results"],
            rag_context=state.get("rag_context", {}),
        ),
    )
    return {"final_itinerary": itinerary}


def _rewrite_followup_qa(state: FollowupQaState) -> Dict[str, Any]:
    rewritten = _run_stage(
        state,
        "followup_rewrite",
        lambda: rewrite_followup_query(
            user_message=state["user_message"],
            parsed_request=state.get("parsed_request"),
            current_itinerary=state.get("final_itinerary"),
            conversation_history=state.get("conversation_history", []),
        ),
    )
    return {"rewritten_message": rewritten}


def _answer_followup_qa(state: FollowupQaState) -> Dict[str, Any]:
    answer = _run_stage(
        state,
        "followup_answer",
        lambda: answer_followup_with_llm(
            parsed_request=state.get("parsed_request", {}),
            execution_plan=state.get("execution_plan", {}),
            tool_results=state.get("tool_results", {}),
            final_itinerary=state.get("final_itinerary", {}),
            conversation_history=state.get("conversation_history", []),
            user_message=state["user_message"],
            rewritten_user_message=state["rewritten_message"],
        ),
    )
    return {"answer": answer}


@lru_cache(maxsize=1)
def get_trip_generation_graph():
    graph = StateGraph(TripGenerationState)
    graph.add_node("route", _noop)
    graph.add_node("rewrite_initial", _rewrite_initial)
    graph.add_node("parse_initial", _parse_initial)
    graph.add_node("rewrite_update", _rewrite_update)
    graph.add_node("update_request", _update_request)
    graph.add_node("plan_trip", _plan_trip)
    graph.add_node("run_tools", _run_tools)
    graph.add_node("retrieve_knowledge", _retrieve_knowledge)
    graph.add_node("generate_itinerary", _generate_itinerary)

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        _route_generation_mode,
        {
            "initial": "rewrite_initial",
            "update": "rewrite_update",
        },
    )
    graph.add_edge("rewrite_initial", "parse_initial")
    graph.add_edge("parse_initial", "plan_trip")
    graph.add_edge("rewrite_update", "update_request")
    graph.add_edge("update_request", "plan_trip")
    graph.add_edge("plan_trip", "run_tools")
    graph.add_edge("run_tools", "retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "generate_itinerary")
    graph.add_edge("generate_itinerary", END)
    return graph.compile()


@lru_cache(maxsize=1)
def get_followup_qa_graph():
    graph = StateGraph(FollowupQaState)
    graph.add_node("rewrite_followup", _rewrite_followup_qa)
    graph.add_node("answer_followup", _answer_followup_qa)
    graph.set_entry_point("rewrite_followup")
    graph.add_edge("rewrite_followup", "answer_followup")
    graph.add_edge("answer_followup", END)
    return graph.compile()


def run_trip_generation_workflow(
    *,
    user_message: str,
    is_update: bool,
    current_parsed_request: Optional[Dict[str, Any]],
    current_itinerary: Optional[Dict[str, Any]],
    previous_tool_results: Optional[Dict[str, Any]],
    conversation_history: List[Dict[str, str]],
    stage_runner: Optional[StageRunner] = None,
) -> TripGenerationState:
    return get_trip_generation_graph().invoke(
        {
            "user_message": user_message,
            "is_update": is_update,
            "current_parsed_request": current_parsed_request,
            "current_itinerary": current_itinerary,
            "previous_tool_results": previous_tool_results,
            "conversation_history": conversation_history,
            "stage_runner": stage_runner,
        }
    )


def run_followup_qa_workflow(
    *,
    user_message: str,
    parsed_request: Dict[str, Any],
    execution_plan: Dict[str, Any],
    tool_results: Dict[str, Any],
    final_itinerary: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    stage_runner: Optional[StageRunner] = None,
) -> FollowupQaState:
    return get_followup_qa_graph().invoke(
        {
            "user_message": user_message,
            "parsed_request": parsed_request,
            "execution_plan": execution_plan,
            "tool_results": tool_results,
            "final_itinerary": final_itinerary,
            "conversation_history": conversation_history,
            "stage_runner": stage_runner,
        }
    )
