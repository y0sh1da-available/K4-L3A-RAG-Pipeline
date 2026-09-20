"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bkai-foundation-models/vietnamese-bi-encoder")
EMBEDDING_DIM = 768

COLLECTION_NAME = "rag_documents"

_ST_MODEL = None


def get_st_model():
    """Tải và cache SentenceTransformer model trên bộ nhớ."""
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
        print(f"Đang tải embedding model: {model_name}...")
        _ST_MODEL = SentenceTransformer(model_name)
    return _ST_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Tạo embeddings cho danh sách texts theo EMBEDDING_PROVIDER."""
    if not texts:
        return []

    provider = os.getenv("EMBEDDING_PROVIDER", EMBEDDING_PROVIDER).lower()

    if provider == "sentence_transformers":
        model = get_st_model()
        vectors = model.encode(texts, batch_size=64, show_progress_bar=False)
        return vectors.tolist()

    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "Chưa tìm thấy GEMINI_API_KEY trong file .env. "
                "Vui lòng điền GEMINI_API_KEY vào file .env trước khi chạy."
            )
        import time
        from google import genai
        client = genai.Client(api_key=api_key)
        model_name = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        batch_size = 100
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            success = False
            for attempt in range(10):
                try:
                    response = client.models.embed_content(
                        model=model_name,
                        contents=batch,
                    )
                    for emb in response.embeddings:
                        all_embeddings.append(emb.values)
                    success = True
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        print(f"\n  [Gemini Rate Limit] Lần {attempt + 1}/10: Đang chờ 40 giây...", flush=True)
                        time.sleep(40)
                    else:
                        raise e

            if not success:
                raise RuntimeError("Không thể lấy embedding sau 10 lần thử do Rate Limit.")

            if len(texts) > batch_size:
                print(f"  -> Đã embed {min(i + batch_size, len(texts))}/{len(texts)} chunks...", flush=True)
                time.sleep(2)
        return all_embeddings

    else:
        raise ValueError(f"Không hỗ trợ provider: {provider}")


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document theo contract."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if path.name.startswith("."):
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue

        doc_type = "legal" if "legal" in path.parts else "news"
        title = path.stem
        url = None

        for line in text.splitlines()[:10]:
            trimmed = line.strip()
            if trimmed.startswith("# "):
                candidate_title = trimmed[2:].strip()
                if candidate_title:
                    title = candidate_title
            elif trimmed.startswith("**Source:**"):
                src_val = trimmed.replace("**Source:**", "").strip()
                if src_val.startswith("http"):
                    url = src_val

        documents.append({
            "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
            "content": text,
            "metadata": {
                "source": path.name,
                "title": title,
                "doc_type": doc_type,
                "url": url,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index theo contract."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for document in documents:
        split_texts = splitter.split_text(document["content"])
        for index, text in enumerate(split_texts):
            clean_text = text.strip()
            if not clean_text:
                continue
            chunks.append({
                "id": f"{document['id']}::chunk-{index}",
                "content": clean_text,
                "metadata": {
                    "source": document["metadata"]["source"],
                    "title": document["metadata"]["title"],
                    "doc_type": document["metadata"]["doc_type"],
                    "url": document["metadata"]["url"],
                    "chunk_index": index,
                },
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    texts = [chunk["content"] for chunk in chunks]
    vectors = embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB an toàn theo batch."""
    if not chunks:
        return

    collection = get_collection()
    batch_size = 250
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[
                {
                    "source": chunk["metadata"]["source"],
                    "title": chunk["metadata"]["title"],
                    "doc_type": chunk["metadata"]["doc_type"],
                    "url": chunk["metadata"]["url"] or "",
                    "chunk_index": chunk["metadata"]["chunk_index"],
                }
                for chunk in batch
            ],
        )


def reset_collection_if_needed():
    """Xóa collection cũ nếu dimension không khớp hoặc muốn làm mới."""
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        col = client.get_collection(COLLECTION_NAME)
        # Kiểm tra dimension của vector đầu tiên nếu có
        sample = col.get(limit=1, include=["embeddings"])
        if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
            existing_dim = len(sample["embeddings"][0])
            if existing_dim != EMBEDDING_DIM:
                print(f"Dimension cũ ({existing_dim}) khác với model mới ({EMBEDDING_DIM}). Đang reset collection...")
                client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index tránh nhân bản dữ liệu."""
    reset_collection_if_needed()

    documents = load_documents()
    print(f"Đã đọc {len(documents)} tài liệu.")
    chunks = chunk_documents(documents)
    print(f"Đã chia thành {len(chunks)} chunks.")

    collection = get_collection()
    existing_data = collection.get(include=[])
    existing_ids = set(existing_data["ids"]) if existing_data and "ids" in existing_data else set()

    chunks_to_index = [c for c in chunks if c["id"] not in existing_ids]

    if not chunks_to_index:
        print(f"Toàn bộ {len(chunks)} chunks đã có sẵn trong ChromaDB (không cần embed lại).")
        return

    print(f"Số chunks mới cần vector hóa và lưu: {len(chunks_to_index)} (đã có {len(existing_ids)} trong DB)")
    batch_size = 200
    for i in range(0, len(chunks_to_index), batch_size):
        batch = chunks_to_index[i : i + batch_size]
        print(f"Đang xử lý lô {i + 1} - {min(i + batch_size, len(chunks_to_index))}...", flush=True)
        embedded_batch = embed_chunks(batch)
        index_to_vectorstore(embedded_batch)

    total_in_db = collection.count()
    print(f"\nThành công! Tổng số chunks trong ChromaDB: {total_in_db}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_pipeline()
