"""
Task 10 — Generation có citation.

Luồng xử lý:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Validate citation.
    6. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal;
không sử dụng kiến thức ngoài context để bịa câu trả lời.
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv

from .task9_retrieval_pipeline import retrieve, _clean_results


load_dotenv()


# =========================================================
# Configuration
# =========================================================

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

SAFE_REFUSAL = (
    "Tôi không thể xác minh thông tin này từ nguồn hiện có."
)


SYSTEM_PROMPT = """
Bạn là một hệ thống RAG grounded.

Chỉ được trả lời dựa trên Context được cung cấp.

Quy tắc bắt buộc:
1. Không sử dụng kiến thức bên ngoài Context.
2. Mọi khẳng định factual phải có citation dạng [Document N].
3. Chỉ citation những Document thực sự hỗ trợ khẳng định đó.
4. Không được tạo citation không tồn tại.
5. Nếu Context không đủ evidence để trả lời câu hỏi,
   hãy trả lời chính xác:
   "Tôi không thể xác minh thông tin này từ nguồn hiện có."
6. Không suy đoán hoặc tự bổ sung thông tin còn thiếu.
""".strip()


# =========================================================
# 1. Lost-in-the-middle reordering
# =========================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Đưa các chunk quan trọng về đầu và cuối context.

    Giả sử chunks ban đầu đã được xếp theo relevance:

        [1, 2, 3, 4, 5]

    Sau reorder:

        [1, 3, 5, 4, 2]

    Như vậy:
        - chunk #1 nằm đầu context
        - chunk #2 nằm cuối context

    Hai chunk quan trọng nhất không bị nằm giữa context.
    """

    if len(chunks) <= 2:
        return list(chunks)

    front = chunks[::2]
    back = chunks[1::2]

    return front + back[::-1]


# =========================================================
# 2. Context formatting
# =========================================================

def format_context(chunks: list[dict]) -> str:
    """
    Convert SearchResult[] thành context cho LLM.

    Input:
        [
            {
                "id": "...",
                "content": "...",
                "score": ...,
                "metadata": {
                    "title": "...",
                    "source": "...",
                    "page": ...
                },
                "retrieval_method": "..."
            }
        ]

    Output:
        [Document 1 | Title: ... | Source: ... | Page: ...]
        nội dung...

        ---

        [Document 2 | ...]
        nội dung...
    """

    parts: list[str] = []

    for index, chunk in enumerate(chunks, start=1):

        metadata = chunk.get("metadata") or {}

        source = (
            metadata.get("source")
            or metadata.get("url")
            or "Unknown source"
        )

        title = (
            metadata.get("title")
            or metadata.get("document_title")
            or source
            or "Untitled"
        )

        page = metadata.get("page")

        content = str(
            chunk.get("content", "")
        ).strip()

        if not content:
            continue

        label = (
            f"[Document {index}"
            f" | Title: {title}"
            f" | Source: {source}"
        )

        if page is not None:
            label += f" | Page: {page}"

        label += "]"

        parts.append(
            f"{label}\n{content}"
        )

    return "\n\n---\n\n".join(parts)


# =========================================================
# 3. Provider dispatch
# =========================================================

def call_llm(
    system_prompt: str,
    user_message: str,
) -> str:
    """
    Gọi LLM provider được cấu hình trong .env.

    Supported:
        openai
        gemini
        anthropic

    .env ví dụ:

        LLM_PROVIDER=openai
        LLM_MODEL=gpt-5-nano
        OPENAI_API_KEY=...

    hoặc:

        LLM_PROVIDER=gemini
        LLM_MODEL=<your-enabled-gemini-model>
        GEMINI_API_KEY=...

    hoặc:

        LLM_PROVIDER=anthropic
        LLM_MODEL=<your-enabled-anthropic-model>
        ANTHROPIC_API_KEY=...
    """

    provider = LLM_PROVIDER

    if not LLM_MODEL:
        raise RuntimeError(
            "LLM_MODEL chưa được cấu hình trong .env"
        )

    # =====================================================
    # OpenAI
    # =====================================================

    if provider == "openai":

        api_key = os.getenv("OPENAI_API_KEY", "")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY chưa được cấu hình."
            )

        try:
            from openai import OpenAI

        except ImportError as exc:
            raise RuntimeError(
                "Thiếu package openai. "
                "Cài bằng: pip install openai"
            ) from exc

        client = OpenAI(
            api_key=api_key,
        )

        response = client.responses.create(
            model=LLM_MODEL,
            instructions=system_prompt,
            input=user_message,
        )

        text = response.output_text

        if not text:
            raise RuntimeError(
                "OpenAI trả về response rỗng."
            )

        return text.strip()

    # =====================================================
    # Gemini
    # =====================================================

    if provider == "gemini":

        api_key = os.getenv("GEMINI_API_KEY", "")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY chưa được cấu hình."
            )

        try:
            from google import genai
            from google.genai import types

        except ImportError as exc:
            raise RuntimeError(
                "Thiếu package google-genai. "
                "Cài bằng: pip install google-genai"
            ) from exc

        client = genai.Client(
            api_key=api_key,
        )

        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            ),
        )

        text = response.text

        if not text:
            raise RuntimeError(
                "Gemini trả về response rỗng."
            )

        return text.strip()

    # =====================================================
    # Anthropic
    # =====================================================

    if provider == "anthropic":

        api_key = os.getenv(
            "ANTHROPIC_API_KEY",
            "",
        )

        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY chưa được cấu hình."
            )

        try:
            import anthropic

        except ImportError as exc:
            raise RuntimeError(
                "Thiếu package anthropic. "
                "Cài bằng: pip install anthropic"
            ) from exc

        client = anthropic.Anthropic(
            api_key=api_key,
        )

        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=1500,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
        )

        text_parts = []

        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)

        text = "".join(text_parts).strip()

        if not text:
            raise RuntimeError(
                "Anthropic trả về response rỗng."
            )

        return text

    # =====================================================
    # Unknown provider
    # =====================================================

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider}. "
        "Supported providers: openai, gemini, anthropic."
    )


# =========================================================
# 4. Citation validation
# =========================================================

def _validate_citations(
    answer: str,
    document_count: int,
) -> bool:
    """
    Kiểm tra citation cơ bản.

    Ví dụ hợp lệ:
        "Hợp đồng phải đáp ứng... [Document 1]"

    Không hợp lệ:
        "Hợp đồng phải đáp ứng..."
            -> không có citation

        "... [Document 99]"
            -> document không tồn tại

    Đây không phải factual entailment checker.
    Nó chỉ là deterministic guardrail.
    """

    if not isinstance(answer, str):
        return False

    if answer.strip() == SAFE_REFUSAL:
        return True

    citation_numbers = re.findall(
        r"\[Document\s+(\d+)\]",
        answer,
        flags=re.IGNORECASE,
    )

    if not citation_numbers:
        return False

    for number in citation_numbers:

        document_number = int(number)

        if (
            document_number < 1
            or document_number > document_count
        ):
            return False

    return True


# =========================================================
# 5. End-to-end generation
# =========================================================

def _get_retrieval_source(chunks: list[dict]) -> str:
    """Contract's hybrid label denotes the primary retrieval route,
    including its single-channel degraded mode; per-result method stays exact.
    """
    if not chunks:
        return "none"
    return "pageindex" if all(c["retrieval_method"] == "pageindex" for c in chunks) else "hybrid"


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    empty = {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return empty
    try:
        sources = _clean_results(retrieve(query.strip(), top_k=top_k), top_k)
    except Exception:
        return empty
    if not sources:
        return empty

    # Public sources remain score-sorted. Reorder only the LLM context.
    reordered = reorder_for_llm(sources)
    context = format_context(reordered)
    result = {"answer": SAFE_REFUSAL, "sources": sources,
              "retrieval_source": _get_retrieval_source(sources)}
    user_message = (
        f"Context:\n\n{context}\n\nQuestion:\n{query.strip()}\n\n"
        "Hãy trả lời chỉ dựa trên Context, dùng citation [Document N]."
    )
    try:
        answer = call_llm(SYSTEM_PROMPT, user_message)
        if not _validate_citations(answer, len(reordered)):
            return result
        # Translate context positions back to public sources positions.
        source_numbers = {source["id"]: n for n, source in enumerate(sources, 1)}
        answer = re.sub(
            r"\[Document\s+(\d+)\]",
            lambda match: f"[Document {source_numbers[reordered[int(match[1]) - 1]['id']]}]",
            answer.strip(), flags=re.IGNORECASE,
        )
        result["answer"] = answer
    except Exception:
        # Provider failure is distinct operationally, but public schema stays fixed.
        return result
    return result


# =========================================================
# Manual test
# =========================================================

if __name__ == "__main__":

    result = generate_with_citation(
        "test query"
    )

    print("\n=== ANSWER ===")
    print(result["answer"])

    print("\n=== RETRIEVAL SOURCE ===")
    print(result["retrieval_source"])

    print("\n=== SOURCES ===")

    for index, source in enumerate(
        result["sources"],
        start=1,
    ):
        print(
            f"\nDocument {index}: "
            f"{source.get('metadata', {})}"
        )