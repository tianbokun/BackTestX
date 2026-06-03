import streamlit as st
import json
import os


_LOCALE_CACHE = {}


def _load(lang: str) -> dict:
    if lang not in _LOCALE_CACHE:
        path = os.path.join(os.path.dirname(__file__), "..", "locales", f"{lang}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                _LOCALE_CACHE[lang] = json.load(f)
        else:
            _LOCALE_CACHE[lang] = {}
    return _LOCALE_CACHE[lang]


def t(key: str, **kwargs) -> str:
    lang = st.session_state.get("_lang", "zh")
    table = _load(lang)
    val = table.get(key)
    if val is None:
        table_zh = _load("zh")
        val = table_zh.get(key, key)
    if kwargs:
        val = val.format(**kwargs)
    return val


def tt(key_zh: str, key_en: str) -> str:
    """Inline zh/en pair — no locale file needed for simple cases."""
    lang = st.session_state.get("_lang", "zh")
    return key_zh if lang == "zh" else key_en


def init_language():
    qp = st.query_params
    lang = qp.get("lang", "zh")
    if lang not in ("zh", "en"):
        lang = "zh"
    if "_lang" not in st.session_state:
        st.session_state._lang = lang


def set_language(lang: str):
    st.session_state._lang = lang
    st.query_params["lang"] = lang



