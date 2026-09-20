"""
Task 8 — PageIndex vectorless fallback.

Flow:
    standardized documents
        -> convert Markdown to PDF if needed
        -> upload to PageIndex
        -> cache source -> document ID
        -> query PageIndex
        -> parse citations / retrieved evidence
        -> SearchResult-like dictionaries

Output SearchResult format:
{
    "id": str,
    "content": str,
    "score": float,
    "metadata": dict,
    "retrieval_method": "pageindex"
}
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

BASE_URL = "https://api.pageindex.ai"

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

CACHE_FILE = (
    Path(__file__).parent.parent
    / "data"
    / "pageindex_document_cache.json"
)

REQUEST_TIMEOUT = 60
PROCESSING_TIMEOUT = 300
POLL_INTERVAL = 3


# =========================================================
# Helpers
# =========================================================

def _headers(json_request: bool = False) -> dict[str, str]:
    """
    Header dùng cho PageIndex API.
    """
    headers = {
        "api_key": PAGEINDEX_API_KEY,
    }

    if json_request:
        headers["Content-Type"] = "application/json"

    return headers


def _file_hash(path: Path) -> str:
    """
    Tính SHA256 của file.

    Dùng để biết file có thay đổi kể từ lần upload trước hay không.
    """
    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def _load_cache() -> dict[str, dict[str, Any]]:
    """
    Đọc cache:

    {
        "/path/to/document.md": {
            "doc_id": "pi-xxx",
            "hash": "...",
            "uploaded_name": "document.pdf"
        }
    }
    """
    if not CACHE_FILE.exists():
        return {}

    try:
        with CACHE_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, dict[str, Any]]) -> None:
    """
    Lưu mapping source -> PageIndex document ID.
    """
    CACHE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CACHE_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            cache,
            f,
            ensure_ascii=False,
            indent=2,
        )


def _request(
    method: str,
    url: str,
    *,
    retries: int = 2,
    timeout: int = REQUEST_TIMEOUT,
    **kwargs: Any,
) -> requests.Response:
    """
    Wrapper requests có:
        - timeout
        - retry
        - xử lý HTTP 429 / 5xx

    Không retry với lỗi 4xx thông thường.
    """

    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = requests.request(
                method,
                url,
                timeout=timeout,
                **kwargs,
            )

            if response.status_code == 429 or response.status_code >= 500:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                    continue

            response.raise_for_status()

            return response

        except requests.RequestException as exc:
            last_error = exc

            if attempt < retries:
                time.sleep(2 ** attempt)
                continue

    raise RuntimeError(
        f"PageIndex request failed: {last_error}"
    )


# =========================================================
# Markdown -> PDF
# =========================================================

def _find_unicode_font() -> str | None:
    """
    Tìm font Unicode phổ biến để PDF hiển thị tiếng Việt.

    Windows / Linux / macOS đều được thử.
    """

    candidates = [
        # Windows
        Path("C:/Windows/Fonts/arial.ttf"),

        # Linux
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),

        # macOS
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    return None


def _markdown_to_pdf(
    markdown_path: Path,
    output_pdf: Path,
) -> None:
    """
    Convert Markdown thành PDF đơn giản.

    Không cố render Markdown 100% như browser.
    Mục tiêu của Task 8 là giữ:
        - heading
        - paragraph
        - logical document structure

    để PageIndex có thể build tree index.
    """

    try:
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import (
            ParagraphStyle,
            getSampleStyleSheet,
        )
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )

    except ImportError as exc:
        raise RuntimeError(
            "Markdown cần convert sang PDF. "
            "Cài reportlab bằng: pip install reportlab"
        ) from exc

    font_name = "Helvetica"

    unicode_font = _find_unicode_font()

    if unicode_font:
        font_name = "PageIndexUnicode"
        pdfmetrics.registerFont(
            TTFont(font_name, unicode_font)
        )

    styles = getSampleStyleSheet()

    normal_style = ParagraphStyle(
        "PI_Normal",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=6,
    )

    heading_styles: dict[int, ParagraphStyle] = {}

    for level in range(1, 7):
        heading_styles[level] = ParagraphStyle(
            f"PI_H{level}",
            parent=styles["Heading1"],
            fontName=font_name,
            fontSize=max(18 - level * 2, 10),
            leading=max(22 - level * 2, 13),
            spaceBefore=10,
            spaceAfter=6,
        )

    text = markdown_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    story = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            story.append(Spacer(1, 6))
            continue

        heading_match = re.match(
            r"^(#{1,6})\s+(.+)$",
            line,
        )

        if heading_match:
            level = len(heading_match.group(1))
            content = heading_match.group(2)

            story.append(
                Paragraph(
                    _escape_reportlab_text(content),
                    heading_styles[level],
                )
            )

            continue

        # loại bớt Markdown syntax cơ bản
        line = re.sub(r"^\s*[-*+]\s+", "• ", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)
        line = re.sub(r"`([^`]*)`", r"\1", line)

        story.append(
            Paragraph(
                _escape_reportlab_text(line),
                normal_style,
            )
        )

    document = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    document.build(story)


def _escape_reportlab_text(text: str) -> str:
    """
    Escape ký tự đặc biệt dùng trong ReportLab Paragraph.
    """
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# =========================================================
# Upload
# =========================================================

def _wait_until_ready(doc_id: str) -> bool:
    """
    Poll PageIndex cho tới khi document:
        completed
    hoặc:
        failed / timeout
    """

    start_time = time.time()

    while time.time() - start_time < PROCESSING_TIMEOUT:
        response = _request(
            "GET",
            f"{BASE_URL}/doc/{doc_id}/metadata",
            headers=_headers(),
        )

        metadata = response.json()

        status = metadata.get("status")

        if status == "completed":
            return True

        if status == "failed":
            return False

        time.sleep(POLL_INTERVAL)

    print(
        f"[PageIndex] Processing timeout: {doc_id}"
    )

    return False


def _upload_pdf(pdf_path: Path) -> str:
    """
    Upload 1 PDF.

    Input:
        Path("abc.pdf")

    Output:
        "pi-abc123..."
    """

    with pdf_path.open("rb") as f:
        response = _request(
            "POST",
            f"{BASE_URL}/doc/",
            headers=_headers(),
            files={
                "file": (
                    pdf_path.name,
                    f,
                    "application/pdf",
                )
            },
        )

    data = response.json()

    doc_id = data.get("doc_id")

    if not doc_id:
        raise RuntimeError(
            f"PageIndex did not return doc_id: {data}"
        )

    return doc_id


def _source_metadata(source_path: Path, documents: list[dict] | None = None) -> dict:
    """Use Task 3/4 Documents or <filename>.metadata.json; never guess doc_type.

    Pass upload_documents(load_documents()) when metadata source resolves to the
    standardized file. Otherwise provide a sidecar containing the original
    metadata dict (or {"metadata": {...}}). Additional metadata is retained.
    """
    matches = []
    for document in documents or []:
        metadata = document.get("metadata", {})
        source = metadata.get("source")
        if source and (str(source) == str(source_path) or
                       Path(str(source)).resolve() == source_path.resolve() or
                       (STANDARDIZED_DIR / str(source)).resolve() == source_path.resolve()):
            matches.append(dict(metadata))
    if len(matches) > 1:
        raise ValueError("Ambiguous source metadata; provide document-level records")
    if matches:
        metadata = matches[0]
    else:
        sidecar = source_path.with_name(source_path.name + ".metadata.json")
        if not sidecar.exists():
            raise ValueError(f"Missing original metadata for {source_path.name}")
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        metadata = data.get("metadata", data)
    if not _valid_metadata(metadata):
        raise ValueError("Original metadata must include source, title, doc_type and url")
    return dict(metadata)


def _valid_metadata(metadata) -> bool:
    return (isinstance(metadata, dict)
            and isinstance(metadata.get("source"), str) and bool(metadata["source"])
            and isinstance(metadata.get("title"), str)
            and metadata.get("doc_type") in {"legal", "news"}
            and "url" in metadata
            and (metadata["url"] is None or isinstance(metadata["url"], str)))


def upload_documents(documents: list[dict] | None = None) -> None:
    """
    Upload tài liệu và lưu document IDs để tái sử dụng.

    Input:
        data/standardized/*.pdf
        data/standardized/*.md
        data/standardized/*.markdown

    Output thực tế:
        data/pageindex_document_cache.json

    Ví dụ:

    {
        ".../law.md": {
            "doc_id": "pi-123",
            "hash": "abc...",
            "uploaded_name": "law.pdf"
        }
    }
    """

    if not PAGEINDEX_API_KEY:
        print(
            "[PageIndex] PAGEINDEX_API_KEY is missing. "
            "Skip PageIndex upload."
        )
        return

    if not STANDARDIZED_DIR.exists():
        print(
            f"[PageIndex] Directory not found: "
            f"{STANDARDIZED_DIR}"
        )
        return

    cache = _load_cache()

    supported_files = [
        path
        for path in STANDARDIZED_DIR.rglob("*")
        if path.suffix.lower()
        in {".pdf", ".md", ".markdown"}
    ]

    if not supported_files:
        print(
            "[PageIndex] No documents found."
        )
        return

    for source_path in supported_files:

        source_key = str(source_path.resolve())
        try:
            current_hash = _file_hash(source_path)
            source_metadata = _source_metadata(source_path, documents)
        except Exception as exc:
            print(f"[PageIndex] Skip {source_path.name}: {exc}")
            continue

        cached = cache.get(source_key)

        # ---------------------------------------------
        # Cache hit
        # ---------------------------------------------
        if (
            cached
            and cached.get("doc_id")
            and cached.get("hash") == current_hash
        ):
            cached["metadata"] = source_metadata
            _save_cache(cache)
            print(
                f"[PageIndex] Cached: "
                f"{source_path.name} "
                f"-> {cached['doc_id']}"
            )
            continue

        try:

            # -----------------------------------------
            # PDF: upload trực tiếp
            # -----------------------------------------
            if source_path.suffix.lower() == ".pdf":

                doc_id = _upload_pdf(source_path)

                uploaded_name = source_path.name

            # -----------------------------------------
            # Markdown: convert PDF tạm
            # -----------------------------------------
            else:

                with tempfile.TemporaryDirectory() as tmp:
                    temp_pdf = (
                        Path(tmp)
                        / f"{source_path.stem}.pdf"
                    )

                    _markdown_to_pdf(
                        source_path,
                        temp_pdf,
                    )

                    doc_id = _upload_pdf(temp_pdf)

                    uploaded_name = temp_pdf.name

            print(
                f"[PageIndex] Uploaded "
                f"{source_path.name} "
                f"-> {doc_id}"
            )

            # -----------------------------------------
            # Chờ PageIndex index xong
            # -----------------------------------------
            ready = _wait_until_ready(doc_id)

            if not ready:
                print(
                    f"[PageIndex] Processing failed: "
                    f"{source_path.name}"
                )
                continue

            # -----------------------------------------
            # Cache
            # -----------------------------------------
            cache[source_key] = {
                "doc_id": doc_id,
                "hash": current_hash,
                "uploaded_name": uploaded_name,
                "source": source_path.name,
                "metadata": source_metadata,
            }

            _save_cache(cache)

        except Exception as exc:
            # PageIndex là fallback.
            # 1 document lỗi không được làm toàn pipeline chết.
            print(
                f"[PageIndex] Failed to upload "
                f"{source_path.name}: {exc}"
            )


# =========================================================
# Retrieval
# =========================================================

def _get_ocr_pages(
    doc_id: str,
) -> list[dict[str, Any]]:
    """
    Lấy page-level OCR/Markdown của document.

    Output ví dụ:
    [
        {
            "page_index": 1,
            "markdown": "..."
        },
        ...
    ]
    """

    response = _request(
        "GET",
        f"{BASE_URL}/doc/{doc_id}/",
        headers=_headers(),
        params={
            "type": "ocr",
            "format": "page",
        },
    )

    data = response.json()

    result = data.get("result", [])

    if isinstance(result, list):
        return result

    return []


def _get_block(
    doc_id: str,
    block_id: str,
) -> dict[str, Any] | None:
    """
    Lấy đúng block được PageIndex citation.
    """

    try:
        response = _request(
            "GET",
            (
                f"{BASE_URL}/doc/"
                f"{doc_id}/block/{block_id}/"
            ),
            headers=_headers(),
        )

        return response.json()

    except Exception:
        return None


def _extract_inline_citations(
    text: str,
) -> list[dict[str, Any]]:
    """
    Fallback nếu API response không chứa citations array.

    Parse citation dạng:

    <cite doc="law.pdf" page="12" block="p12_text_3"/>

    hoặc page-only:

    <cite doc="law.pdf" page="12"/>
    """

    pattern = re.compile(
        r'<cite\s+'
        r'doc="(?P<doc>[^"]+)"\s+'
        r'page="(?P<page>\d+)"'
        r'(?:\s+block="(?P<block>[^"]+)")?'
        r'\s*/?>'
    )

    citations = []

    for match in pattern.finditer(text):
        citations.append(
            {
                "doc": match.group("doc"),
                "page": int(match.group("page")),
                "block": match.group("block"),
            }
        )

    return citations


def _pageindex_search(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Vectorless retrieval dùng PageIndex.

    INPUT
    -----
    query:
        câu hỏi tự nhiên.

    Ví dụ:
        "Điều kiện để hợp đồng lao động có hiệu lực?"

    top_k:
        số SearchResult tối đa trả về.

    OUTPUT
    ------
    [
        {
            "id": "...",
            "content": "...",
            "score": 1.0,
            "metadata": {
                "source": "...",
                "doc_id": "...",
                "page": 12,
                "block_id": "..."
            },
            "retrieval_method": "pageindex"
        }
    ]
    """

    if not query.strip():
        return []

    if top_k <= 0:
        return []

    if not PAGEINDEX_API_KEY:
        print(
            "[PageIndex] PAGEINDEX_API_KEY missing. "
            "Fallback disabled."
        )
        return []

    cache = _load_cache()

    if not cache:
        print(
            "[PageIndex] No cached documents. "
            "Run upload_documents() first."
        )
        return []

    cache = {key: item for key, item in cache.items()
             if isinstance(item, dict) and _valid_metadata(item.get("metadata"))}

    doc_ids = [
        item["doc_id"]
        for item in cache.values()
        if item.get("doc_id")
    ]

    if not doc_ids:
        return []

    # Reverse lookup:
    # doc_id -> cached metadata
    doc_by_id = {
        item["doc_id"]: item
        for item in cache.values()
        if item.get("doc_id")
    }

    # filename -> doc_id
    doc_name_to_id: dict[str, str] = {}

    ambiguous_names: set[str] = set()
    for item in cache.values():
        doc_id = item.get("doc_id")
        if not doc_id:
            continue
        for name in (item.get("uploaded_name"), item.get("source")):
            if not name:
                continue
            name = Path(str(name)).name
            if name in doc_name_to_id and doc_name_to_id[name] != doc_id:
                ambiguous_names.add(name)
            doc_name_to_id[name] = doc_id
    for name in ambiguous_names:
        doc_name_to_id.pop(name, None)

    try:
        # ---------------------------------------------
        # PageIndex reasoning-based retrieval
        # ---------------------------------------------
        response = _request(
            "POST",
            f"{BASE_URL}/chat/completions",
            headers=_headers(json_request=True),
            json={
                "doc_id": doc_ids,
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
                "stream": False,
                "temperature": 0.0,
                "enable_citations": True,
            },
        )

        data = response.json()

    except Exception as exc:
        print(
            f"[PageIndex] Search failed: {exc}"
        )
        return []

    # ---------------------------------------------
    # Assistant answer
    # ---------------------------------------------
    try:
        answer_text = (
            data["choices"][0]
            ["message"]["content"]
        )

    except (KeyError, IndexError, TypeError):
        answer_text = ""

    # ---------------------------------------------
    # Citations
    # ---------------------------------------------
    citations = data.get("citations")

    if not citations:
        # Một số API shape có thể đặt citations trong message
        try:
            citations = (
                data["choices"][0]
                ["message"]
                .get("citations")
            )
        except (KeyError, IndexError, TypeError):
            citations = None

    if not citations:
        citations = _extract_inline_citations(
            answer_text
        )

    if not citations:
        return []

    results: list[dict[str, Any]] = []

    ocr_cache: dict[
        str,
        list[dict[str, Any]]
    ] = {}

    seen: set[str] = set()

    # ---------------------------------------------
    # citation -> SearchResult
    # ---------------------------------------------
    for citation in citations:

        if not isinstance(citation, dict):
            continue

        if len(results) >= top_k:
            break

        doc_id = (
            citation.get("doc_id")
            or citation.get("document_id")
        )

        doc_name = (
            citation.get("doc")
            or citation.get("document")
            or citation.get("document_name")
        )

        if not doc_id and doc_name:
            doc_id = doc_name_to_id.get(
                Path(str(doc_name)).name
            )

        # Nếu chỉ có 1 document, có thể suy ra.
        if not doc_id and not doc_name and len(doc_ids) == 1:
            doc_id = doc_ids[0]

        if doc_id not in doc_by_id:
            continue

        try:
            page = int(
                citation.get("page")
                or citation.get("page_index")
                or 0
            )
        except (TypeError, ValueError):
            page = 0

        block_id = (
            citation.get("block")
            or citation.get("block_id")
        )

        content = ""
        bbox = citation.get("bbox")

        # -----------------------------------------
        # Best case: citation có block
        # -----------------------------------------
        if block_id:

            block = _get_block(
                doc_id,
                str(block_id),
            )

            if block:
                content = block.get("text") or ""

                page = int(
                    block.get("page", page or 0)
                )

                bbox = (
                    block.get("bbox")
                    or bbox
                )

        # -----------------------------------------
        # Fallback: lấy cả page
        # -----------------------------------------
        if not content and page:

            if doc_id not in ocr_cache:
                try:
                    ocr_cache[doc_id] = (
                        _get_ocr_pages(doc_id)
                    )
                except Exception:
                    ocr_cache[doc_id] = []

            for page_item in ocr_cache[doc_id]:

                if (
                    page_item.get("page_index")
                    == page
                ):
                    content = str(
                        page_item.get(
                            "markdown",
                            "",
                        )
                    )
                    break

        if not content.strip():
            continue

        cached_doc = doc_by_id.get(
            doc_id,
            {},
        )

        # Identify actual evidence: multiple missing blocks on one page collapse.
        if not content or not isinstance(content, str):
            continue
        stable_source = cached_doc["metadata"]["source"]
        evidence_key = json.dumps([stable_source, page, content.strip()], ensure_ascii=False)
        evidence_id = "pageindex:" + hashlib.sha256(evidence_key.encode()).hexdigest()
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        rank = len(results)

        # PageIndex Chat API không đảm bảo relevance score.
        # Vì vậy gán score giảm theo rank.
        score = 1.0 / (rank + 1)

        result = {
            "id": evidence_id,
            "content": content.strip(),
            "score": score,
            "metadata": {
                **cached_doc["metadata"],
                "doc_id": doc_id,
                "page": page,
                "block_id": block_id,
                "bbox": bbox,
            },
            "retrieval_method": "pageindex",
        }

        results.append(result)

    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Provider/cache/response failures must not escape the fallback boundary."""
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []
    try:
        return _pageindex_search(query.strip(), top_k)
    except Exception:
        print("[PageIndex] Fallback unavailable or invalid response.")
        return []


# =========================================================
# Manual test
# =========================================================

if __name__ == "__main__":

    upload_documents()

    print("\n--- TEST SEARCH ---")

    test_results = pageindex_search(
        "What are the main legal requirements?",
        top_k=5,
    )

    for index, result in enumerate(
        test_results,
        start=1,
    ):
        print(f"\nResult #{index}")
        print("ID:", result["id"])
        print("Score:", result["score"])
        print("Metadata:", result["metadata"])
        print(
            "Content:",
            result["content"][:500],
        )