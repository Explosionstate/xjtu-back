from __future__ import annotations

from typing import Any, Callable, TypedDict

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover
    END = None  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]


AGENT_SYSTEM_PROMPTS: dict[str, str] = {
    "student-growth": "你是学生成长助手，回答要具体、可执行，兼顾学习与生活建议。",
    "teacher-assistant": "你是教师助教助手，回答要聚焦教学设计、课堂互动和评估反馈。",
    "counselor-ideology": "你是辅导员思政助手，回答要注重价值引导、沟通方式和管理可行性。",
    "risk-warning": "你是学情预警助手，回答要给出风险分级、证据和优先处置建议。",
    "report-assistant": "你是学情报告助手，输出结构化报告，包含结论、依据、建议。",
    "policy-qa": "你是思政知识问答助手，回答需准确、可追溯并给出政策依据。",
}


class ChatGraphState(TypedDict):
    question: str
    agent_key: str
    sensitive_words: list[str]
    blocked_word: str
    blocked_stage: str
    error: str
    system_instruction: str
    profile_context: str
    contexts: list[str]
    sources: list[dict[str, Any]]
    answer: str
    detect_fn: Callable[[str, list[str]], str | None]
    load_profile_fn: Callable[[str], str]
    retrieve_fn: Callable[[str], tuple[list[str], list[dict[str, Any]]]]
    generate_fn: Callable[[str, list[str], str], str]


def _normalize_input(state: ChatGraphState) -> ChatGraphState:
    state["question"] = (state.get("question") or "").strip()
    state["agent_key"] = (state.get("agent_key") or "").strip().lower()
    return state


def _input_guard(state: ChatGraphState) -> ChatGraphState:
    detect_fn = state["detect_fn"]
    blocked = detect_fn(state.get("question", ""), state.get("sensitive_words", []))
    if blocked:
        state["blocked_word"] = blocked
        state["blocked_stage"] = "input"
        state["error"] = "INPUT_BLOCKED"
    return state


def _route_prompt(state: ChatGraphState) -> ChatGraphState:
    default_prompt = "你是西交 AI 智能体，请基于知识库给出准确、简洁、结构化的回答。"
    state["system_instruction"] = AGENT_SYSTEM_PROMPTS.get(
        state.get("agent_key", ""),
        default_prompt,
    )
    return state


def _profile_enrich(state: ChatGraphState) -> ChatGraphState:
    if state.get("error"):
        return state
    loader = state["load_profile_fn"]
    state["profile_context"] = loader(state.get("question", ""))
    return state


def _retrieval(state: ChatGraphState) -> ChatGraphState:
    if state.get("error"):
        return state
    retrieve = state["retrieve_fn"]
    contexts, sources = retrieve(state.get("question", ""))
    state["contexts"] = contexts
    state["sources"] = sources
    return state


def _generation(state: ChatGraphState) -> ChatGraphState:
    if state.get("error"):
        return state
    profile = state.get("profile_context", "")
    contexts = list(state.get("contexts", []))
    if profile:
        contexts = [profile] + contexts
    generate = state["generate_fn"]
    state["answer"] = generate(
        state.get("question", ""), contexts, state.get("system_instruction", "")
    )
    return state


def _output_guard(state: ChatGraphState) -> ChatGraphState:
    if state.get("error"):
        return state
    detect_fn = state["detect_fn"]
    blocked = detect_fn(state.get("answer", ""), state.get("sensitive_words", []))
    if blocked:
        state["blocked_word"] = blocked
        state["blocked_stage"] = "output"
        state["error"] = "OUTPUT_BLOCKED"
    return state


def _build_graph():
    if StateGraph is None:
        return None
    graph = StateGraph(ChatGraphState)
    graph.add_node("normalize", _normalize_input)
    graph.add_node("input_guard", _input_guard)
    graph.add_node("route_prompt", _route_prompt)
    graph.add_node("profile", _profile_enrich)
    graph.add_node("retrieval", _retrieval)
    graph.add_node("generation", _generation)
    graph.add_node("output_guard", _output_guard)
    graph.set_entry_point("normalize")
    graph.add_edge("normalize", "input_guard")
    graph.add_edge("input_guard", "route_prompt")
    graph.add_edge("route_prompt", "profile")
    graph.add_edge("profile", "retrieval")
    graph.add_edge("retrieval", "generation")
    graph.add_edge("generation", "output_guard")
    graph.add_edge("output_guard", END)
    return graph.compile()


def run_chat_workflow_graph(
    *,
    question: str,
    agent_key: str | None,
    sensitive_words: list[str],
    detect_fn: Callable[[str, list[str]], str | None],
    load_profile_fn: Callable[[str], str],
    retrieve_fn: Callable[[str], tuple[list[str], list[dict[str, Any]]]],
    generate_fn: Callable[[str, list[str], str], str],
) -> dict[str, Any]:
    state: ChatGraphState = {
        "question": question or "",
        "agent_key": (agent_key or "").strip().lower(),
        "sensitive_words": sensitive_words,
        "blocked_word": "",
        "blocked_stage": "",
        "error": "",
        "system_instruction": "",
        "profile_context": "",
        "contexts": [],
        "sources": [],
        "answer": "",
        "detect_fn": detect_fn,
        "load_profile_fn": load_profile_fn,
        "retrieve_fn": retrieve_fn,
        "generate_fn": generate_fn,
    }

    app = _build_graph()
    if app is None:
        # Runtime fallback when langgraph is unavailable.
        normalized_question = (question or "").strip()
        blocked = detect_fn(normalized_question, sensitive_words)
        if blocked:
            return {
                "error": "INPUT_BLOCKED",
                "blocked_word": blocked,
                "blocked_stage": "input",
                "question": normalized_question,
                "system_instruction": "",
                "contexts": [],
                "sources": [],
                "answer": "",
            }
        instruction = AGENT_SYSTEM_PROMPTS.get(
            (agent_key or "").strip().lower(),
            "你是西交 AI 智能体，请基于知识库给出准确、简洁、结构化的回答。",
        )
        profile = load_profile_fn(normalized_question)
        contexts, sources = retrieve_fn(normalized_question)
        merged_contexts = ([profile] if profile else []) + list(contexts)
        answer = generate_fn(normalized_question, merged_contexts, instruction)
        blocked_output = detect_fn(answer, sensitive_words)
        if blocked_output:
            return {
                "error": "OUTPUT_BLOCKED",
                "blocked_word": blocked_output,
                "blocked_stage": "output",
                "question": normalized_question,
                "system_instruction": instruction,
                "contexts": contexts,
                "sources": sources,
                "answer": answer,
                "profile_context": profile,
            }
        return {
            "error": "",
            "blocked_word": "",
            "blocked_stage": "",
            "question": normalized_question,
            "system_instruction": instruction,
            "contexts": contexts,
            "sources": sources,
            "answer": answer,
            "profile_context": profile,
        }

    result = app.invoke(state)
    return dict(result)
