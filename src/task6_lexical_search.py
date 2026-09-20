"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""

import re
import numpy as np
from rank_bm25 import BM25L

CORPUS: list[dict] = []


def _tokenize(text: str) -> list[str]:
    """Tách từ chuẩn xác: xử lý dấu gạch nối (Trưng-Trắc -> Trưng, Trắc) và loại bỏ dấu câu."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    tokenized = [_tokenize(item["content"]) for item in corpus]
    return BM25L(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    global CORPUS
    if not CORPUS:
        from .task4_chunking_indexing import chunk_documents, load_documents
        CORPUS.extend(chunk_documents(load_documents()))

    if not CORPUS:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    bm25 = build_bm25_index(CORPUS)
    scores = bm25.get_scores(tokens)
    indices = np.argsort(scores)[::-1]

    results: list[dict] = []
    for index in indices:
        if len(results) >= top_k:
            break
        # Chỉ lấy chunk có độ tương đồng dương
        if scores[index] <= 0:
            continue

        item = CORPUS[index]
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(scores[index]),
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })

    return results


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for result in lexical_search("Số Đỏ", top_k=3):
        print(f"[{result['retrieval_method']}] score={result['score']:.4f} | {result['metadata']['source']}::chunk-{result['metadata']['chunk_index']}")
        print(f"Content snippet: {result['content'][:120]}...\n")
