# Agent Lessons

## 1. Streamlit 无限 rerun 问题

**症状**：widget（如 selectbox）选中后浏览器标签页持续闪烁，必须手动刷新。

**根因**：`st.selectbox` 等输入 widget 的值变化会触发一次隐式 rerun。handler 里再调 `st.rerun()` 形成无限循环。

**教训**：widget 值变化后绝不在同一次 run 内调 `st.rerun()`。正确的做法是用 `st.button` 做显式确认，或使用 `on_change` 回调。

---

## 2. ID 冲突问题

**症状**：`StreamlitDuplicateElementId` 错误。

**根因**：多个元素（如 button）使用相同 label 和参数时，Streamlit 会生成相同 auto-generated ID。

**教训**：任何时候创建 button、slider、selectbox 等 widget，**必须**传唯一的 `key` 参数，尤其当不同模块/函数可能创建相同 label 的元素时。

---

## 3. 未捕获异常导致整页不渲染

**症状**：页面某模块渲染函数无错误提示、无内容，连后续其他模块也一起消失。

**根因**：函数中对外部调用（如 fetch_history）未包 `try/except`，抛出的异常穿透到顶层，终止了整页渲染流程。

**教训**：任何发起网络请求/IO 的调用（尤其是 `fetch_history`、`_cached_fetch` 等被 `@st.cache_data` 包装的函数），调用侧必须包 `try/except` 防止异常上抛。Streamlit 页面渲染链中的每一个环节都应该保证不抛异常，否则后续所有内容都不会显示。
