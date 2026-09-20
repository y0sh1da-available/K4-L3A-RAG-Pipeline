"""Task 9: fuse once; threshold always uses original dense cosine."""
from copy import deepcopy
from math import isfinite
import logging

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
logger = logging.getLogger(__name__)


def _clean_results(results, top_k, method=None):
    """Copy, validate, sort and deduplicate without changing source metadata."""
    unique = {}
    for item in results or []:
        if not isinstance(item, dict):
            continue
        if not isinstance(item.get("id"), str) or not item["id"]:
            continue
        if not isinstance(item.get("content"), str) or not item["content"].strip():
            continue
        if not isinstance(item.get("metadata"), dict):
            continue
        try:
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if not isfinite(score):
            continue
        result = deepcopy(item)
        result["score"] = score
        if method:
            result["retrieval_method"] = method
        if result.get("retrieval_method") not in {"dense", "bm25", "hybrid", "pageindex"}:
            continue
        previous = unique.get(result["id"])
        if previous is None or score > previous["score"]:
            unique[result["id"]] = result
    return sorted(unique.values(), key=lambda item: item["score"], reverse=True)[:max(0, top_k)]


def _safe_search(search, query, top_k, method):
    try:
        return _clean_results(search(query, top_k=top_k), top_k, method)
    except Exception:
        logger.warning("%s retrieval failed", method)
        return []


def retrieve(query: str, top_k: int = DEFAULT_TOP_K,
             score_threshold: float = SCORE_THRESHOLD,
             use_reranking: bool = True) -> list[dict]:
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []
    query = query.strip()
    dense = _safe_search(semantic_search, query, top_k * 2, "dense")
    # Capture BEFORE fusion: a teammate's reranker may mutate its inputs.
    best_dense_score = max((r["score"] for r in dense), default=float("-inf"))
    if use_reranking:
        sparse = _safe_search(lexical_search, query, top_k * 2, "bm25")
        try:
            hybrid = _clean_results(
                rerank_rrf(deepcopy([dense, sparse]), top_k=top_k), top_k, "hybrid"
            ) if dense or sparse else []
        except Exception:
            logger.warning("RRF failed; retaining single-channel evidence")
            # Never mix BM25 and cosine scores or perform a second fusion.
            hybrid = _clean_results(dense or sparse, top_k)
    else:
        hybrid = dense[:top_k]
    if not dense or best_dense_score < score_threshold:
        fallback = _safe_search(pageindex_search, query, top_k, "pageindex")
        if fallback:
            return fallback
    return hybrid
