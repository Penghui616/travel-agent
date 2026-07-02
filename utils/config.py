import os
from typing import Optional


def get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value not in (None, ""):
        return value

    try:
        import streamlit as st

        if name in st.secrets:
            secret_value = st.secrets[name]
            return str(secret_value) if secret_value is not None else default
    except Exception:
        pass

    return default


def get_required_setting(name: str) -> str:
    value = get_setting(name)
    if not value:
        raise ValueError(f"请在 Streamlit Secrets 或 .env 中配置 {name}")
    return value
