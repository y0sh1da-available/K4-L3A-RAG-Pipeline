"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    query_vector = embed_texts([query])[0]
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results: list[dict] = []
    if response and response.get("ids") and response["ids"][0]:
        for item_id, content, metadata, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            meta = dict(metadata)
            if meta.get("url") == "":
                meta["url"] = None
            if "chunk_index" in meta and not isinstance(meta["chunk_index"], int):
                meta["chunk_index"] = int(meta["chunk_index"])

            # Cosine similarity = 1.0 - cosine distance
            score = float(max(0.0, min(1.0, 1.0 - distance)))
            results.append({
                "id": item_id,
                "content": content,
                "score": score,
                "metadata": meta,
                "retrieval_method": "dense",
            })

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for result in semantic_search("test query", top_k=3):
        print(f"[{result['retrieval_method']}] score={result['score']:.4f} | {result['metadata']['source']}::chunk-{result['metadata']['chunk_index']}")
        print(f"Content snippet: {result['content'][:120]}...\n")
