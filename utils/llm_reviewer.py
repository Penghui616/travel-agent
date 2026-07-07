import json
import os
from copy import deepcopy
from typing import Any, Dict, List

from dotenv import load_dotenv

from utils.config import get_setting
from utils.langchain_llm import LangChainChatClient, get_langchain_chat_client
from utils.llm_itinerary import extract_json_from_text, postprocess_itinerary
from utils.token_usage import record_token_usage

load_dotenv()

MAX_REPAIR_ATTEMPTS = 2


REVIEW_REPAIR_PROMPT = """
你是一个旅行行程 Reviewer Agent。
你会收到：
1. 当前解析后的用户需求
2. 当前执行计划
3. 当前工具结果
4. 当前生成的行程 JSON
5. 校验器发现的问题列表

你的任务是修复这个行程，让它通过校验。
必须遵守：
- 只输出合法 JSON，不要输出解释，不要输出 markdown。
- 保留原有行程中合理的部分，只修复有问题的地方。
- `days` 数组长度必须严格等于 `parsed_request.days`。
- 同一个地点 `name` 不能重复。
- 每一天至少 3 个 item。
- 默认每天需要覆盖上午、下午、晚上；如果用户明确说习惯下午出门，则第一段行程必须从 12:00 以后开始。
- `source_links` 如果有就保留并规范化；没有也可以为空，不需要为每个 item 强制补来源链接
- 不要删除用户明确要求的核心偏好。
"""


def get_client() -> LangChainChatClient:
    return get_langchain_chat_client()


def get_model_name() -> str:
    return get_setting("ZHIPU_MODEL", "glm-4-flash") or "glm-4-flash"


def _normalize_name(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def _parse_start_hour(time_range: str) -> int | None:
    if not isinstance(time_range, str) or "-" not in time_range:
        return None
    start = time_range.split("-", 1)[0].strip()
    if ":" not in start:
        return None
    hour = start.split(":", 1)[0]
    return int(hour) if hour.isdigit() else None


def _prefers_afternoon_start(parsed_request: Dict[str, Any]) -> bool:
    special_requirements = (parsed_request.get("special_requirements", "") or "").strip()
    return any(
        keyword in special_requirements
        for keyword in [
            "下午出门",
            "下午开始",
            "中午后出门",
            "晚点出门",
            "不想早起",
            "12:00 以后",
        ]
    )


def enrich_source_links(itinerary: Dict[str, Any], city: str) -> Dict[str, Any]:
    cloned = deepcopy(itinerary)
    for day in cloned.get("days", []):
        for item in day.get("items", []):
            source_links = item.get("source_links", [])
            if isinstance(source_links, list) and source_links:
                normalized_links = []
                for link in source_links:
                    if isinstance(link, dict) and link.get("url"):
                        normalized_links.append(
                            {
                                "label": link.get("label", "来源"),
                                "url": link["url"],
                            }
                        )
                    elif isinstance(link, str) and link.strip():
                        normalized_links.append({"label": "来源", "url": link.strip()})
                if normalized_links:
                    item["source_links"] = normalized_links
                    continue
            item["source_links"] = []
    return cloned


def validate_itinerary_for_review(
    itinerary: Dict[str, Any],
    parsed_request: Dict[str, Any],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    expected_days = int(parsed_request.get("days", 0) or 0)
    days = itinerary.get("days", [])

    if expected_days and len(days) != expected_days:
        issues.append(
            {
                "code": "day_count_mismatch",
                "message": f"行程天数不匹配：需求是 {expected_days} 天，结果生成了 {len(days)} 天。",
            }
        )

    seen_names = set()
    for day_index, day in enumerate(days, start=1):
        items = day.get("items", [])
        if len(items) < 3:
            issues.append(
                {
                    "code": "too_few_items",
                    "day": day_index,
                    "message": f"第 {day_index} 天安排过少，当前只有 {len(items)} 个 item。",
                }
            )

        time_slots = {"morning": False, "afternoon": False, "evening": False}
        for item in items:
            name = item.get("name", "").strip()
            normalized = _normalize_name(name)
            if normalized:
                if normalized in seen_names:
                    issues.append(
                        {
                            "code": "duplicate_place",
                            "day": day_index,
                            "place": name,
                            "message": f"发现重复地点：{name}",
                        }
                    )
                seen_names.add(normalized)

            hour = _parse_start_hour(item.get("time", ""))
            if hour is not None:
                if hour < 12:
                    time_slots["morning"] = True
                elif hour < 18:
                    time_slots["afternoon"] = True
                else:
                    time_slots["evening"] = True

        if _prefers_afternoon_start(parsed_request):
            if items:
                first_hour = _parse_start_hour(items[0].get("time", ""))
                if first_hour is not None and first_hour < 12:
                    issues.append(
                        {
                            "code": "afternoon_start_violation",
                            "day": day_index,
                            "message": f"第 {day_index} 天没有按下午出门要求安排，第一段开始时间早于 12:00。",
                        }
                    )
        else:
            for slot, label in [("morning", "上午"), ("afternoon", "下午"), ("evening", "晚上")]:
                if not time_slots[slot]:
                    issues.append(
                        {
                            "code": "missing_time_slot",
                            "day": day_index,
                            "slot": slot,
                            "message": f"第 {day_index} 天缺少{label}安排。",
                        }
                    )

    return issues


def format_review_issues(issues: List[Dict[str, Any]]) -> List[str]:
    return [issue.get("message", "未知校验问题") for issue in issues]


def repair_itinerary_with_llm(
    parsed_request: Dict[str, Any],
    execution_plan: Dict[str, Any],
    tool_results: Dict[str, Any],
    itinerary: Dict[str, Any],
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = {
        "parsed_request": parsed_request,
        "execution_plan": execution_plan,
        "tool_results": tool_results,
        "itinerary": itinerary,
        "validation_issues": issues,
    }

    model_name = get_model_name()
    response = get_client().chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {"role": "system", "content": REVIEW_REPAIR_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    record_token_usage("reviewer_repair", response, model_name)

    content = response.choices[0].message.content
    return extract_json_from_text(content)


def _apply_local_reviewer_fixes(
    itinerary: Dict[str, Any],
    parsed_request: Dict[str, Any],
    tool_results: Dict[str, Any],
) -> Dict[str, Any]:
    city = parsed_request.get("city", "")
    locally_fixed = enrich_source_links(itinerary, city)
    return postprocess_itinerary(locally_fixed, parsed_request, tool_results)


def review_and_repair_itinerary(
    parsed_request: Dict[str, Any],
    execution_plan: Dict[str, Any],
    tool_results: Dict[str, Any],
    itinerary: Dict[str, Any],
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
) -> Dict[str, Any]:
    current_itinerary = postprocess_itinerary(itinerary, parsed_request, tool_results)
    issues = validate_itinerary_for_review(current_itinerary, parsed_request)

    if issues:
        current_itinerary = _apply_local_reviewer_fixes(
            current_itinerary,
            parsed_request,
            tool_results,
        )
        issues = validate_itinerary_for_review(current_itinerary, parsed_request)

    repair_count = 0
    while issues and repair_count < max_repair_attempts:
        try:
            repaired = repair_itinerary_with_llm(
                parsed_request=parsed_request,
                execution_plan=execution_plan,
                tool_results=tool_results,
                itinerary=current_itinerary,
                issues=issues,
            )
        except Exception as exc:
            issues.append(
                {
                    "code": "review_repair_parse_failed",
                    "message": f"Reviewer 修复阶段返回了无法解析的 JSON：{exc}",
                }
            )
            break
        repair_count += 1
        current_itinerary = _apply_local_reviewer_fixes(
            repaired,
            parsed_request,
            tool_results,
        )
        issues = validate_itinerary_for_review(current_itinerary, parsed_request)

    return {
        "itinerary": current_itinerary,
        "passed": not issues,
        "repair_count": repair_count,
        "issues": format_review_issues(issues),
        "raw_issues": issues,
    }
