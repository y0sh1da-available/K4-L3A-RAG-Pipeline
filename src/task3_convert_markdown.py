"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

import json
from pathlib import Path
from markitdown import MarkItDown


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_legal_docs() -> None:
    # TODO:Convert PDF/DOCX vào standardized/legal. 
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print(f"Warning: Thư mục {legal_dir} không tồn tại.")
        return

    converter = MarkItDown()

    for path in legal_dir.iterdir():
        if path.suffix.lower() in {".pdf", ".doc", ".docx"}:
            output_file = output_dir / f"{path.stem}.md"
            if output_file.exists():
                print(f"Skipped (already exists): {output_file.name}")
                continue

            try:
                print(f"Converting PDF/DOCX: {path.name}...")
                result = converter.convert(str(path))

                if result.text_content and result.text_content.strip():
                    output_file.write_text(result.text_content, encoding="utf-8")
                    print(f" Saved: {output_file.name}")
                else:
                    print(f" Warning: File {path.name} convert ra nội dung rỗng!")
            except Exception as e:
                print(f" Error converting {path.name}: {e}")


def convert_news_articles() -> None:
    # TODO: Convert JSON vào standardized/news.
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print(f"Warning: Thư mục {news_dir} không tồn tại.")
        return

    for path in news_dir.glob("*.json"):
        output_file = output_dir / f"{path.stem}.md"

        if output_file.exists():
            print(f"Skipped (already exists): {output_file.name}")
            continue

        try:
            print(f"Converting JSON: {path.name}...")
            data = json.loads(path.read_text(encoding="utf-8"))

            header = (
                f"# {data.get('title', 'Untitled')}\n\n"
                f"**Source:** {data.get('url', '')}\n\n"
                f"**Crawled:** {data.get('date_crawled', '')}\n\n---\n\n"
            )

            content = data.get("content_markdown", "")
            full_markdown = header + content

            if full_markdown.strip():
                output_file.write_text(full_markdown, encoding="utf-8")
                print(f" Saved: {output_file.name}")
            else:
                print(f" Warning: File {path.name} không có nội dung!")
        except Exception as e:
            print(f" Error converting {path.name}: {e}")


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()

