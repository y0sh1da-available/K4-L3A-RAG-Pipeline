* [ ]  Danh Sách Thành Viên & Phân Công Nhiệm Vụ (Group Project)

## 1. Thông Tin Chung

* **Tên nhóm:** y0sh1da
* **Chủ đề dự án:** Sách và Tri thức (Literature & Knowledge RAG)
* **Nhánh tích hợp chính:** `develop`

---

## 2. Danh Sách Thành Viên

### Thành viên 1: Lộc

* **Họ và tên:** Trần Đức Lộc
* **Mã học viên:** 2A202602431
* **Vai trò:** Vai trò 1 — Data & Ingestion Specialist
* **Nhánh làm việc:** `feature/loc`
* **Phần việc phụ trách:**
  - Task 1: Thu thập tài liệu gốc (`task1_collect_legal_docs.py`)
  - Task 2: Crawl và xuất bài viết JSON (`task2_crawl_news.py`)
  - Task 3: Chuẩn hóa dữ liệu sang Markdown (`task3_convert_markdown.py`)
  - Đóng góp 8 câu hỏi Q&A vào `golden_dataset.json`
  - Báo cáo cá nhân: `reports/2A202602431-loc.md`

---

### Thành viên 2: Cương

* **Họ và tên:** Đặng Hữu Cương
* **Mã học viên:** 2A202602572
* **Vai trò:** Vai trò 2 — Search Engine Core Specialist
* **Nhánh làm việc:** `feature/cuong`
* **Phần việc phụ trách:**
  - Task 4: Chunking, Gemini Embedding & ChromaDB Indexing (`task4_chunking_indexing.py`)
  - Task 5: Semantic Search qua ChromaDB (`task5_semantic_search.py`)
  - Task 6: Lexical Search BM25 (`task6_lexical_search.py`)
  - Task 7: RRF Reranking (`task7_reranking.py`)
  - Đóng góp 7 câu hỏi Q&A vào `golden_dataset.json`
  - Báo cáo cá nhân: `reports/2A202602572-Cuong.md`

---

### Thành viên 3: Đức

* **Họ và tên:** *(Sẽ điền sau)*
* **Mã học viên:** *(Sẽ điền sau)*
* **Vai trò:** Vai trò 3 — Pipeline, Generation & UI Lead
* **Nhánh làm việc:** `feature/duc`
* **Phần việc phụ trách:**
  - Task 8: PageIndex / Vectorless fallback handler (`task8_pageindex_vectorless.py`)
  - Task 9: Tích hợp Retrieval Pipeline hoàn chỉnh (`task9_retrieval_pipeline.py`)
  - Task 10: Generation có Citation và Safe refusal (`task10_generation.py`)
  - Streamlit Chatbot UI: Giao diện hiển thị nguồn và câu trả lời (`app.py`)
  - Đánh giá RAGAS (4 metric) & hoàn thiện báo cáo nhóm `RESULT.md`
  - Báo cáo cá nhân: `reports/<student-id>-duc.md`

---

## 3. Quy Trình Phối Hợp Git

1. Mỗi thành viên checkout vào nhánh riêng của mình (`feature/<tên>`) để code.
2. Kiểm tra test hợp đồng trước khi commit: `pytest tests/test_contracts.py -v`.
3. Khi hoàn thành task, commit và tạo Pull Request (PR) merge vào nhánh `develop`.
4. Khi toàn bộ pipeline hoàn tất và kiểm thử pass 100%, trưởng nhóm merge `develop` vào `main` để nộp bài.
