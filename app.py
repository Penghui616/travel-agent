from contextlib import contextmanager
from time import perf_counter
from typing import Any, Dict, Iterator

import streamlit as st

from utils.llm_followup import (
    should_update_trip,
)
from utils.langgraph_workflow import (
    run_followup_qa_workflow,
    run_trip_generation_workflow,
)
from utils.token_usage import (
    get_token_usage_records,
    reset_token_usage,
    summarize_token_usage,
)
from utils.user_memory import (
    load_user_memory,
    memory_to_display,
    reset_user_memory,
    update_user_memory_from_interaction,
)


TIMING_STAGE_LABELS = {
    "initial_rewrite": "首轮 Query Rewriting",
    "parse_request": "需求解析 LLM",
    "followup_rewrite": "追问 Query Rewriting",
    "update_request": "更新需求 LLM",
    "deterministic_plan": "本地任务计划",
    "tools": "工具调用",
    "rag_retrieval": "RAG 知识检索",
    "itinerary_llm": "行程生成（快速/LLM）",
    "followup_answer": "追问回答 LLM",
}

TOKEN_STAGE_LABELS = {
    "query_rewrite": "Query Rewriting",
    "parse_request": "需求解析",
    "update_request": "需求更新",
    "itinerary_generation": "行程生成",
    "json_repair": "JSON 修复",
    "followup_answer": "追问回答",
    "reviewer_repair": "Reviewer 修复",
}

PIPELINE_PROGRESS_TITLES = {
    "initial_trip": "正在生成完整旅行方案",
    "update_trip": "正在根据补充需求调整行程",
    "followup_qa": "正在结合当前方案回答问题",
}

PIPELINE_TOTAL_STEPS = {
    "initial_trip": 6,
    "update_trip": 6,
    "followup_qa": 2,
}


DEFAULT_ASSISTANT_MESSAGE = (
    "你好，我可以帮你规划旅行，也可以基于已有方案继续修改。\n\n"
    "你可以直接说：\n"
    "- 去重庆玩三天\n"
    "- 再加两天\n"
    "- 第一天轻松一点\n"
    "- 不想去博物馆\n"
    "- 我习惯下午出门"
)


st.set_page_config(
    page_title="旅行规划 Agent",
    page_icon="✈️",
    layout="wide",
)


def init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "raw_user_input": "",
        "rewritten_user_input": "",
        "parsed_request": None,
        "execution_plan": None,
        "tool_results": None,
        "final_itinerary": None,
        "last_warnings": [],
        "query_rewrite_history": [],
        "last_pipeline_timings": None,
        "timing_history": [],
        "last_token_usage": None,
        "token_usage_history": [],
        "user_memory": load_user_memory(),
        "conversation_history": [
            {"role": "assistant", "content": DEFAULT_ASSISTANT_MESSAGE}
        ],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_conversation() -> None:
    st.session_state["raw_user_input"] = ""
    st.session_state["rewritten_user_input"] = ""
    st.session_state["parsed_request"] = None
    st.session_state["execution_plan"] = None
    st.session_state["tool_results"] = None
    st.session_state["final_itinerary"] = None
    st.session_state["last_warnings"] = []
    st.session_state["query_rewrite_history"] = []
    st.session_state["last_pipeline_timings"] = None
    st.session_state["timing_history"] = []
    st.session_state["last_token_usage"] = None
    st.session_state["token_usage_history"] = []
    st.session_state["conversation_history"] = [
        {"role": "assistant", "content": DEFAULT_ASSISTANT_MESSAGE}
    ]


class ProgressReporter:
    def __init__(self, pipeline_type: str) -> None:
        self.pipeline_type = pipeline_type
        self.total_steps = max(PIPELINE_TOTAL_STEPS.get(pipeline_type, 1), 1)
        self.completed_steps = 0
        title = PIPELINE_PROGRESS_TITLES.get(pipeline_type, "正在处理")

        self.container = st.container(border=True)
        with self.container:
            st.markdown("#### 实时进度")
            self.progress_bar = st.progress(0)
            self.status_box = st.empty()
            self.detail_box = st.empty()

        self.status_box.info(title)
        self.detail_box.caption("准备开始...")

    def _percent(self) -> int:
        return min(100, int(self.completed_steps / self.total_steps * 100))

    def start(self, stage: str) -> None:
        label = TIMING_STAGE_LABELS.get(stage, stage)
        self.progress_bar.progress(self._percent())
        self.status_box.info(f"正在处理：{label}")
        self.detail_box.caption(f"当前阶段：{label}")

    def finish(self, stage: str, elapsed_seconds: float) -> None:
        self.completed_steps = min(self.completed_steps + 1, self.total_steps)
        label = TIMING_STAGE_LABELS.get(stage, stage)
        self.progress_bar.progress(self._percent())
        self.status_box.success(f"已完成：{label}")
        self.detail_box.caption(f"{label} 用时 {elapsed_seconds:.2f} 秒")

    def fail(self, stage: str, error: Exception) -> None:
        label = TIMING_STAGE_LABELS.get(stage, stage)
        self.status_box.error(f"{label} 失败：{error}")
        self.detail_box.caption("流程已中断，请查看错误信息。")

    def complete(self) -> None:
        self.completed_steps = self.total_steps
        self.progress_bar.progress(100)
        self.status_box.success("处理完成，正在展示结果。")
        self.detail_box.caption("所有阶段已完成。")


@contextmanager
def timed_stage(
    timings: Dict[str, float],
    stage: str,
    progress: ProgressReporter | None = None,
) -> Iterator[None]:
    start_time = perf_counter()
    failed = False
    if progress is not None:
        progress.start(stage)
    try:
        yield
    except Exception as exc:
        failed = True
        if progress is not None:
            progress.fail(stage, exc)
        raise
    finally:
        elapsed_seconds = round(perf_counter() - start_time, 2)
        timings[stage] = elapsed_seconds
        if progress is not None and not failed:
            progress.finish(stage, elapsed_seconds)


def save_pipeline_timings(
    pipeline_type: str,
    timings: Dict[str, float],
    total_start_time: float,
) -> None:
    record = {
        "type": pipeline_type,
        "total_seconds": round(perf_counter() - total_start_time, 2),
        "stages": dict(timings),
    }
    st.session_state["last_pipeline_timings"] = record
    history = st.session_state.get("timing_history", [])
    history.append(record)
    st.session_state["timing_history"] = history[-10:]


def save_token_usage_record(pipeline_type: str) -> None:
    records = get_token_usage_records()
    summary = summarize_token_usage(records)
    record = {
        "type": pipeline_type,
        "summary": summary,
        "records": records,
    }
    st.session_state["last_token_usage"] = record
    history = st.session_state.get("token_usage_history", [])
    history.append(record)
    st.session_state["token_usage_history"] = history[-10:]


def render_itinerary_card(itinerary: Dict[str, Any]) -> None:
    st.markdown("## 最终行程")
    st.markdown(f"# {itinerary.get('title', '旅行行程')}")

    summary = itinerary.get("summary", "")
    if summary:
        st.write(summary)

    tips = itinerary.get("important_tips", [])
    if tips:
        st.markdown("### 重要提醒")
        for tip in tips:
            st.info(tip)

    days = itinerary.get("days", [])
    if not days:
        st.warning("当前没有可展示的行程数据。")
        return

    tab_titles = [f"DAY {day.get('day', index + 1)}" for index, day in enumerate(days)]
    tabs = st.tabs(tab_titles)

    for tab, day in zip(tabs, days):
        with tab:
            day_num = day.get("day", "")
            theme = day.get("theme", "")
            route_summary = day.get("route_summary", "")

            st.markdown(f"## DAY {day_num}｜{theme}")
            if route_summary:
                st.caption(route_summary)

            items = day.get("items", [])
            for idx, item in enumerate(items, start=1):
                name = item.get("name", "未命名地点")
                time_range = item.get("time", "")
                category = item.get("category", "其他")
                description = item.get("description", "")
                transport = item.get("transport_to_next", "")
                source_links = item.get("source_links", [])

                with st.container(border=True):
                    col_left, col_right = st.columns([1, 5])

                    with col_left:
                        st.markdown(f"### {idx}")
                        st.markdown(f"**{category}**")

                    with col_right:
                        st.markdown(f"### {name}")
                        if time_range:
                            st.markdown(f"**时间：** {time_range}")
                        if description:
                            st.write(description)
                        if transport:
                            st.caption(f"➡️ 下一站：{transport}")

                        if isinstance(source_links, list) and source_links:
                            link_fragments = []
                            for link in source_links:
                                if isinstance(link, dict) and link.get("url"):
                                    label = link.get("label", "来源")
                                    link_fragments.append(f"[{label}]({link['url']})")
                                elif isinstance(link, str) and link.strip():
                                    link_fragments.append(f"[来源]({link.strip()})")
                            if link_fragments:
                                st.markdown("来源：" + " | ".join(link_fragments))

            day_tips = day.get("day_tips", [])
            if day_tips:
                st.markdown("### 当日建议")
                for tip in day_tips:
                    st.warning(tip)

    st.markdown("---")
    st.markdown("## 综合建议")

    hotel_area = itinerary.get("hotel_area_suggestion", "")
    weather_advice = itinerary.get("weather_advice", "")
    transport_advice = itinerary.get("transport_advice", "")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 住宿建议")
        st.write(hotel_area or "暂无")

    with col2:
        st.markdown("### 天气建议")
        st.write(weather_advice or "暂无")

    with col3:
        st.markdown("### 交通建议")
        st.write(transport_advice or "暂无")


def build_assistant_summary(
    itinerary: Dict[str, Any],
    parsed_request: Dict[str, Any],
    is_update: bool,
) -> str:
    city = parsed_request.get("city") or "目的地"
    days = parsed_request.get("days") or "?"
    summary = itinerary.get("summary", "").strip()
    title = itinerary.get("title", "").strip()
    hotel_area = itinerary.get("hotel_area_suggestion", "").strip()

    intro = "我已经按你的补充更新了行程。" if is_update else "行程已经生成好了。"
    parts = [intro, f"当前方案：{city} {days} 天。"]

    if title:
        parts.append(f"标题：{title}")
    if summary:
        parts.append(summary)
    if hotel_area:
        parts.append(f"建议住宿区域：{hotel_area}")

    parts.append("如果你还想继续改，我可以基于当前方案接着调整。")
    return "\n\n".join(parts)


def add_query_rewrite_record(stage: str, raw: str, rewritten: str) -> None:
    st.session_state["query_rewrite_history"].append(
        {
            "stage": stage,
            "raw": raw,
            "rewritten": rewritten,
        }
    )


def run_trip_pipeline(user_message: str, is_update: bool) -> None:
    reset_token_usage()
    timings: Dict[str, float] = {}
    total_start_time = perf_counter()
    pipeline_type = "update_trip" if is_update else "initial_trip"
    progress = ProgressReporter(pipeline_type)
    history = st.session_state["conversation_history"]
    success = False

    try:
        def stage_runner(stage: str, fn):
            with timed_stage(timings, stage, progress):
                return fn()

        workflow_result = run_trip_generation_workflow(
            user_message=user_message,
            is_update=is_update,
            current_parsed_request=st.session_state["parsed_request"],
            current_itinerary=st.session_state["final_itinerary"],
            previous_tool_results=st.session_state["tool_results"],
            conversation_history=history,
            stage_runner=stage_runner,
        )

        parsed_request = workflow_result["parsed_request"]
        execution_plan = workflow_result["execution_plan"]
        tool_results = workflow_result["tool_results"]
        final_itinerary = workflow_result["final_itinerary"]
        rewritten_message = workflow_result.get("rewritten_message", user_message)

        if not is_update and not st.session_state["raw_user_input"]:
            st.session_state["raw_user_input"] = user_message
            st.session_state["rewritten_user_input"] = rewritten_message

        add_query_rewrite_record(
            workflow_result.get("rewrite_record_stage", "initial_parse"),
            user_message,
            rewritten_message,
        )

        st.session_state["parsed_request"] = parsed_request
        st.session_state["execution_plan"] = execution_plan
        st.session_state["tool_results"] = tool_results
        st.session_state["final_itinerary"] = final_itinerary
        st.session_state["user_memory"] = update_user_memory_from_interaction(
            user_message,
            parsed_request,
        )
        st.session_state["last_warnings"] = []

        assistant_message = build_assistant_summary(
            itinerary=final_itinerary,
            parsed_request=parsed_request,
            is_update=is_update,
        )
        st.session_state["conversation_history"].append(
            {"role": "assistant", "content": assistant_message}
        )
        success = True
    finally:
        if success:
            progress.complete()
        save_pipeline_timings(pipeline_type, timings, total_start_time)
        save_token_usage_record(pipeline_type)


def handle_user_message(user_message: str) -> None:
    clean_message = user_message.strip()
    if not clean_message:
        return

    st.session_state["conversation_history"].append(
        {"role": "user", "content": clean_message}
    )

    has_existing_trip = st.session_state["parsed_request"] is not None
    if not has_existing_trip:
        with st.spinner("我在理解你的需求并生成完整行程..."):
            run_trip_pipeline(clean_message, is_update=False)
        return

    if should_update_trip(clean_message):
        with st.spinner("我在根据你的补充调整行程..."):
            run_trip_pipeline(clean_message, is_update=True)
        return

    timings: Dict[str, float] = {}
    total_start_time = perf_counter()
    reset_token_usage()
    progress = ProgressReporter("followup_qa")
    success = False
    try:
        def stage_runner(stage: str, fn):
            with timed_stage(timings, stage, progress):
                return fn()

        qa_result = run_followup_qa_workflow(
            user_message=clean_message,
            parsed_request=st.session_state["parsed_request"] or {},
            execution_plan=st.session_state["execution_plan"] or {},
            tool_results=st.session_state["tool_results"] or {},
            final_itinerary=st.session_state["final_itinerary"] or {},
            conversation_history=st.session_state["conversation_history"],
            stage_runner=stage_runner,
        )
        rewritten_message = qa_result.get("rewritten_message", clean_message)
        add_query_rewrite_record("followup_qa", clean_message, rewritten_message)
        st.session_state["conversation_history"].append(
            {"role": "assistant", "content": qa_result["answer"]}
        )
        success = True
    finally:
        if success:
            progress.complete()
        save_pipeline_timings("followup_qa", timings, total_start_time)
        save_token_usage_record("followup_qa")


init_session_state()

st.title("✈️ 旅行规划 Agent")
st.caption(
    "支持多轮对话：先做 Query Rewriting，再结合工具结果生成旅行方案。"
)

_, top_col2, top_col3 = st.columns([4, 1, 1])
with top_col2:
    if st.button("清空记忆", use_container_width=True):
        st.session_state["user_memory"] = reset_user_memory()
        st.rerun()
with top_col3:
    if st.button("清空对话", use_container_width=True):
        reset_conversation()
        st.rerun()

st.subheader("对话区")
chat_container = st.container()
with chat_container:
    for message in st.session_state["conversation_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

prompt = st.chat_input("告诉我你的旅行需求，或继续补充怎么修改当前方案")
if prompt:
    handle_user_message(prompt)
    st.rerun()

st.markdown("---")

if st.session_state["final_itinerary"]:
    render_itinerary_card(st.session_state["final_itinerary"])
else:
    st.info("先在上面的对话框里告诉我你的旅行需求，我会直接开始规划。")

st.markdown("---")
st.subheader("调试信息")

with st.expander("原始首轮输入", expanded=False):
    if st.session_state["raw_user_input"]:
        st.write(st.session_state["raw_user_input"])
    else:
        st.info("当前还没有首轮输入。")

with st.expander("Query Rewriting 结果", expanded=False):
    if st.session_state["rewritten_user_input"]:
        st.markdown("**首轮重写**")
        st.write(st.session_state["rewritten_user_input"])
    if st.session_state["query_rewrite_history"]:
        st.markdown("**最近几次重写**")
        st.json(st.session_state["query_rewrite_history"][-5:])
    if (
        not st.session_state["rewritten_user_input"]
        and not st.session_state["query_rewrite_history"]
    ):
        st.info("当前还没有 Query Rewriting 结果。")

with st.expander("耗时统计", expanded=False):
    timing_record = st.session_state.get("last_pipeline_timings")
    if timing_record:
        st.caption(
            f"最近一次流程：{timing_record.get('type', 'unknown')}，"
            f"总耗时：{timing_record.get('total_seconds', 0)} 秒"
        )
        timing_rows = [
            {
                "阶段": TIMING_STAGE_LABELS.get(stage, stage),
                "耗时（秒）": elapsed,
            }
            for stage, elapsed in timing_record.get("stages", {}).items()
        ]
        if timing_rows:
            st.table(timing_rows)
        else:
            st.info("这次流程还没有记录到具体阶段耗时。")
    else:
        st.info("当前还没有耗时统计。")

with st.expander("Token 消耗统计", expanded=False):
    token_record = st.session_state.get("last_token_usage")
    if token_record:
        summary = token_record.get("summary", {})
        st.caption(
            f"最近一次流程：{token_record.get('type', 'unknown')}，"
            f"LLM 调用次数：{summary.get('call_count', 0)}，"
            f"总 token：{summary.get('total_tokens', 0)}"
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("输入 token", summary.get("prompt_tokens", 0))
        col2.metric("输出 token", summary.get("completion_tokens", 0))
        col3.metric("总 token", summary.get("total_tokens", 0))

        token_rows = [
            {
                "阶段": TOKEN_STAGE_LABELS.get(record.get("stage"), record.get("stage")),
                "模型": record.get("model", ""),
                "输入 token": record.get("prompt_tokens", 0),
                "输出 token": record.get("completion_tokens", 0),
                "总 token": record.get("total_tokens", 0),
            }
            for record in token_record.get("records", [])
        ]
        if token_rows:
            st.table(token_rows)
        else:
            st.info("这次流程没有产生新的 LLM token 消耗，可能是命中了缓存或没有调用模型。")
    else:
        st.info("当前还没有 Token 消耗统计。")

with st.expander("长期记忆", expanded=False):
    st.caption("长期记忆会自动保存用户偏好，并在新行程中作为软约束注入。")
    st.json(memory_to_display(st.session_state.get("user_memory")))

with st.expander("解析后的旅行需求", expanded=False):
    if st.session_state["parsed_request"] is not None:
        st.json(st.session_state["parsed_request"])
    else:
        st.info("当前还没有解析结果。")

with st.expander("Agent 任务计划", expanded=False):
    if st.session_state["execution_plan"] is not None:
        st.json(st.session_state["execution_plan"])
    else:
        st.info("当前还没有任务计划。")

with st.expander("工具执行结果", expanded=False):
    if st.session_state["tool_results"] is not None:
        st.json(st.session_state["tool_results"])
    else:
        st.info("当前还没有工具结果。")

with st.expander("最终行程 JSON", expanded=False):
    if st.session_state["final_itinerary"] is not None:
        st.json(st.session_state["final_itinerary"])
    else:
        st.info("当前还没有最终行程。")
