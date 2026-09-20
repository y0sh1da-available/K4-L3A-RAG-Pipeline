"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Tải ít nhất 3 PDF/DOCX từ nguồn công khai."""
    # TODO: Có thể tải thủ công hoặc dùng requests.
    import requests
    
    sources = {
         "tuoitre.pdf": "https://dlib.ptit.edu.vn/bitstream/HVCNBCVT/2285/1/Tuoi%20tre%20dang%20gia%20bao%20nhieu.pdf",
         "vnsuluoc.pdf": "https://cvdvn.net/wp-content/uploads/2018/03/viet-nam-su-luoc-tran-trong-kim1.pdf",
         "sodo.pdf": "https://medialib.qlgd.edu.vn/Uploads/THU_VIEN/shn/2/1005/UserFiles/SachMoi.Net-so-do-vu-trong-phung-cfc411df-f9f8-4b31-825c-0ba17d5d2c12.pdf",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    for filename, url in sources.items():
         response = requests.get(url, timeout=30, headers=headers, verify=False)
         response.raise_for_status()
         (DATA_DIR / filename).write_bytes(response.content)
    


if __name__ == "__main__":
    setup_directory()
    download_documents()
