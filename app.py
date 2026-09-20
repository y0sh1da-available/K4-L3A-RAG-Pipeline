"""Run from the project root: streamlit run app.py."""
from __future__ import annotations

import logging
import math
import re
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

# Load environment BEFORE importing Task 10 (which reads it at import time).
load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")
logger = logging.getLogger(__name__)
SAFE_REFUSAL = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

st.markdown(
    """
    <style>
    .block-container {max-width: 960px; padding-top: 2rem; padding-bottom: 5rem;}
    [data-testid="stChatMessage"] {border-radius: 16px; padding: 1.1rem;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.15);}
    </style>
    """, unsafe_allow_html=True,
)


def safe_url(value) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = urlparse(value)
        return value if parsed.scheme in {"http", "https"} and parsed.netloc else None
    except ValueError:
        return None


def score_label(source: dict) -> str:
    method = source.get("retrieval_method", "unknown")
    label = {"dense": "Cosine", "bm25": "BM25", "hybrid": "RRF",
             "pageindex": "Điểm theo thứ hạng"}.get(method, "Score")
    try:
        score = float(source.get("score"))
        value = f"{score:.6f}" if math.isfinite(score) else "N/A"
    except (TypeError, ValueError):
        value = "N/A"
    return f"{method} · {label}: {value}"


def render_assistant(message: dict) -> None:
    answer = message["content"]
    st.markdown(answer)
    if message.get("error"):
        st.warning(message["error"])
    sources = message.get("sources", [])
    route = message.get("retrieval_source", "none")
    route_label = {"hybrid": "Truy xuất chính", "pageindex": "PageIndex fallback",
                   "none": "Không có nguồn"}.get(route, route)
    st.caption(f"{route_label} · {len(sources)} đoạn tài liệu")
    if not sources:
        return

    cited = {int(n) for n in re.findall(r"\[Document\s+(\d+)\]", answer, re.I)}
    st.caption("Mở nguồn để xem nội dung. Score là điểm truy xuất, không phải xác suất câu trả lời đúng.")
    # Never sort/filter here: Document N must remain sources[N - 1].
    for number, source in enumerate(sources, 1):
        metadata = source.get("metadata") or {}
        title = str(metadata.get("title") or metadata.get("source") or "Tài liệu")
        marker = " · Đã trích dẫn" if number in cited else " · Nguồn truy xuất"
        with st.expander(f"[Document {number}] {title}{marker}"):
            st.text(f"Nguồn: {metadata.get('source') or 'Không có thông tin'}")
            if metadata.get("page") is not None:
                st.text(f"Trang: {metadata['page']}")
            st.caption(score_label(source))
            url = safe_url(metadata.get("url")) or safe_url(metadata.get("source"))
            if url:
                st.link_button("Mở tài liệu gốc", url)
            # Source content is data; do not render embedded HTML or Markdown links.
            st.text(source.get("content", ""))


def ask_pipeline(query: str, top_k: int) -> dict:
    message = {"role": "assistant", "content": SAFE_REFUSAL,
               "sources": [], "retrieval_source": "none"}
    try:
        # Lazy import allows the UI to open even when a teammate's module is missing.
        from src.task10_generation import generate_with_citation
        result = generate_with_citation(query, top_k=top_k)
        answer = result["answer"]
        sources = result["sources"]
        route = result["retrieval_source"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Invalid answer")
        if not isinstance(sources, list) or route not in {"hybrid", "pageindex", "none"}:
            raise ValueError("Invalid GenerationResult")
        if any(not isinstance(s, dict) or not isinstance(s.get("metadata"), dict)
               or not isinstance(s.get("content"), str) for s in sources):
            raise ValueError("Invalid source")
        message.update(content=answer, sources=sources, retrieval_source=route)
    except Exception:
        logger.exception("RAG pipeline failed")
        message["error"] = "Chưa xử lý được yêu cầu. Kiểm tra cấu hình và log trong terminal rồi thử lại."
    return message


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("💬 RAG Chatbot")
    st.caption("Tra cứu tài liệu pháp luật và tin tức trong bộ dữ liệu của nhóm.")
    top_k = st.slider("Số đoạn tài liệu tối đa", 3, 10, 5)
    st.caption("Áp dụng cho câu hỏi tiếp theo.")
    if st.button("Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Câu trả lời dựa trên tài liệu đã được chuẩn bị và lập chỉ mục.")

st.title("RAG Chatbot")
st.caption("Hỏi về tài liệu của nhóm và kiểm tra nguồn ngay dưới mỗi câu trả lời.")

if not st.session_state.messages:
    st.info("Bạn muốn tìm hiểu điều gì? Nhập câu hỏi vào ô chat bên dưới.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant(message)
        else:
            st.markdown(message["content"])

query = st.chat_input("Nhập câu hỏi về tài liệu...")
if query and query.strip():
    query = query.strip()
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm tài liệu và tạo câu trả lời..."):
            assistant_message = ask_pipeline(query, top_k)
        # Save before rendering, so reruns retain the answer and its sources.
        st.session_state.messages.append(assistant_message)
        render_assistant(assistant_message)
