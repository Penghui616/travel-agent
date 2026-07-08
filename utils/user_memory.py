import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from utils.config import get_setting


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_MEMORY_FILE = DATA_DIR / "user_memory.json"

AFTERNOON_KEYWORDS = [
    "下午出门",
    "下午开始",
    "中午后出门",
    "晚点出门",
    "不想早起",
    "睡到自然醒",
    "习惯下午出门",
]
FULL_DAY_KEYWORDS = ["从早到晚", "上午出门", "上午开始", "早上出门", "全天", "安排满"]
KNOWN_AVOID_PLACES = ["博物馆", "展馆", "美术馆", "寺庙", "爬山", "酒吧", "商场", "购物", "夜店"]
MEMORY_NOTE_LIMIT = 8


def memory_enabled() -> bool:
    value = str(get_setting("TRAVEL_AGENT_ENABLE_MEMORY", "1") or "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _memory_file() -> Path:
    custom_path = (get_setting("TRAVEL_AGENT_MEMORY_FILE", "") or "").strip()
    return Path(custom_path) if custom_path else DEFAULT_MEMORY_FILE


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _empty_memory() -> Dict[str, Any]:
    return {
        "version": 1,
        "preferences": [],
        "avoid_places": [],
        "start_time_preference": "",
        "transport_preference": "",
        "budget": "",
        "travel_group": "",
        "notes": [],
        "update_count": 0,
        "updated_at": "",
    }


def _dedupe(items: List[Any]) -> List[str]:
    deduped = []
    seen = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def load_user_memory() -> Dict[str, Any]:
    memory = _empty_memory()
    if not memory_enabled():
        return memory

    path = _memory_file()
    if not path.exists():
        return memory

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return memory

    if isinstance(loaded, dict):
        memory.update(loaded)
    memory["preferences"] = _dedupe(memory.get("preferences", []))
    memory["avoid_places"] = _dedupe(memory.get("avoid_places", []))
    memory["notes"] = _dedupe(memory.get("notes", []))[-MEMORY_NOTE_LIMIT:]
    return memory


def save_user_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    if not memory_enabled():
        return _empty_memory()

    path = _memory_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = _empty_memory()
    cleaned.update(memory or {})
    cleaned["preferences"] = _dedupe(cleaned.get("preferences", []))
    cleaned["avoid_places"] = _dedupe(cleaned.get("avoid_places", []))
    cleaned["notes"] = _dedupe(cleaned.get("notes", []))[-MEMORY_NOTE_LIMIT:]
    cleaned["updated_at"] = _now_iso()
    path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    return cleaned


def reset_user_memory() -> Dict[str, Any]:
    path = _memory_file()
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
    return _empty_memory()


def _append_special_requirement(original: str, addition: str) -> str:
    original = (original or "").strip()
    addition = (addition or "").strip()
    if not addition:
        return original
    if not original:
        return addition
    if addition in original:
        return original
    return f"{original}；{addition}"


def _extract_avoid_places(text: str, parsed_request: Dict[str, Any]) -> List[str]:
    avoid_places: List[str] = []
    constraints = parsed_request.get("hard_constraints", {}) or {}
    avoid_places.extend(constraints.get("avoid_places_global", []) or [])

    for place in KNOWN_AVOID_PLACES:
        if any(pattern in text for pattern in [f"不想去{place}", f"不去{place}", f"避开{place}", f"不喜欢{place}"]):
            avoid_places.append(place)

    for match in re.finditer(r"(?:不想去|不去|不要去|别安排|避开|不喜欢)([^，。；,.!！?？\s]{1,16})", text):
        phrase = match.group(1).strip()
        phrase = re.sub(r"(了|吧|这些|这类|这种|地方|景点)$", "", phrase).strip()
        if phrase and len(phrase) <= 12:
            avoid_places.append(phrase)

    return _dedupe(avoid_places)


def extract_memory_from_interaction(
    user_message: str,
    parsed_request: Dict[str, Any],
) -> Dict[str, Any]:
    text = f"{user_message} {parsed_request.get('special_requirements', '')}"
    extracted = _empty_memory()

    preferences = parsed_request.get("preferences", []) or []
    extracted["preferences"] = _dedupe(preferences)
    extracted["avoid_places"] = _extract_avoid_places(text, parsed_request)

    if any(keyword in text for keyword in AFTERNOON_KEYWORDS):
        extracted["start_time_preference"] = "afternoon"
    elif any(keyword in text for keyword in FULL_DAY_KEYWORDS):
        extracted["start_time_preference"] = "full_day"

    for field in ["transport_preference", "budget", "travel_group"]:
        value = str(parsed_request.get(field, "") or "").strip()
        if value and value not in {"都可以", "不限", "无"}:
            extracted[field] = value

    notes = []
    if extracted["start_time_preference"] == "afternoon":
        notes.append("用户习惯下午出门。")
    if extracted["avoid_places"]:
        notes.append(f"用户希望避开：{'、'.join(extracted['avoid_places'][:6])}。")
    if extracted["preferences"]:
        notes.append(f"用户偏好：{'、'.join(extracted['preferences'][:6])}。")
    extracted["notes"] = notes
    return extracted


def merge_user_memory(
    current_memory: Dict[str, Any],
    new_memory: Dict[str, Any],
) -> Dict[str, Any]:
    merged = _empty_memory()
    merged.update(current_memory or {})
    changed = False

    for field in ["preferences", "avoid_places", "notes"]:
        combined = _dedupe((merged.get(field, []) or []) + (new_memory.get(field, []) or []))
        if combined != merged.get(field, []):
            merged[field] = combined[-MEMORY_NOTE_LIMIT:] if field == "notes" else combined
            changed = True

    for field in ["start_time_preference", "transport_preference", "budget", "travel_group"]:
        value = str(new_memory.get(field, "") or "").strip()
        if value and value != merged.get(field):
            merged[field] = value
            changed = True

    if changed:
        merged["update_count"] = int(merged.get("update_count", 0) or 0) + 1
        return save_user_memory(merged)
    return merged


def update_user_memory_from_interaction(
    user_message: str,
    parsed_request: Dict[str, Any],
) -> Dict[str, Any]:
    if not memory_enabled():
        return _empty_memory()
    current_memory = load_user_memory()
    new_memory = extract_memory_from_interaction(user_message, parsed_request)
    return merge_user_memory(current_memory, new_memory)


def apply_memory_to_request(
    parsed_request: Dict[str, Any],
    memory: Dict[str, Any] | None = None,
    user_message: str = "",
) -> Dict[str, Any]:
    if not memory_enabled():
        return parsed_request

    memory = memory or load_user_memory()
    updated = deepcopy(parsed_request)
    text = f"{user_message} {updated.get('special_requirements', '')}"
    special = updated.get("special_requirements", "")

    memory_preferences = _dedupe(memory.get("preferences", []))
    current_preferences = _dedupe(updated.get("preferences", []) or [])
    if memory_preferences and not current_preferences:
        updated["preferences"] = memory_preferences[:4]
        special = _append_special_requirement(
            special,
            f"长期记忆：用户偏好 {('、'.join(memory_preferences[:4]))}，可优先兼顾。",
        )

    avoid_places = _dedupe(memory.get("avoid_places", []))
    if avoid_places:
        special = _append_special_requirement(
            special,
            f"长期记忆：用户希望避开 {('、'.join(avoid_places[:6]))}。",
        )
        constraints = deepcopy(updated.get("hard_constraints", {}) or {})
        constraints.setdefault("avoid_places_global", [])
        constraints["avoid_places_global"] = _dedupe(
            constraints.get("avoid_places_global", []) + avoid_places
        )
        updated["hard_constraints"] = constraints

    start_time = memory.get("start_time_preference", "")
    user_overrides_afternoon = any(keyword in text for keyword in FULL_DAY_KEYWORDS)
    user_mentions_afternoon = any(keyword in text for keyword in AFTERNOON_KEYWORDS)
    if start_time == "afternoon" and not user_overrides_afternoon and not user_mentions_afternoon:
        special = _append_special_requirement(
            special,
            "长期记忆：用户习惯下午出门，尽量把每天第一段行程安排在 12:00 以后。",
        )

    if start_time == "full_day" and not user_mentions_afternoon:
        special = _append_special_requirement(
            special,
            "长期记忆：用户偏好从早到晚安排充实。",
        )

    transport = str(memory.get("transport_preference", "") or "").strip()
    if transport and (not updated.get("transport_preference") or updated.get("transport_preference") == "都可以"):
        updated["transport_preference"] = transport

    for field in ["budget", "travel_group"]:
        value = str(memory.get(field, "") or "").strip()
        if value and not updated.get(field):
            updated[field] = value

    updated["special_requirements"] = special
    return updated


def memory_to_display(memory: Dict[str, Any] | None = None) -> Dict[str, Any]:
    memory = memory or load_user_memory()
    return {
        "偏好": memory.get("preferences", []),
        "避开": memory.get("avoid_places", []),
        "出门时间": memory.get("start_time_preference", "") or "未记录",
        "交通偏好": memory.get("transport_preference", "") or "未记录",
        "预算": memory.get("budget", "") or "未记录",
        "出行人群": memory.get("travel_group", "") or "未记录",
        "记忆笔记": memory.get("notes", []),
        "更新时间": memory.get("updated_at", "") or "暂无",
        "更新次数": memory.get("update_count", 0),
    }
