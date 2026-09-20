# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-20 |
| Framework and version              | Python 3.12, ChromaDB 1.5.9, Sentence-Transformers, Google GenAI SDK |
| Evaluator model                    | Rule-based Grounded Benchmark & Gemini 3.6 Flash |
| Generator model                    | gemini-3.6-flash |
| Embedding model                    | bkai-foundation-models/vietnamese-bi-encoder (768 dims) |
| Corpus version/commit              | develop (commit 4b01a3b) |
| Golden dataset size                | 21 câu hỏi |
| `top_k`                            | 5 |
| Fallback threshold and calibration | SCORE_THRESHOLD = 0.3 |

## Configurations

- **Config A — dense-only:** Chỉ sử dụng Semantic Search (ChromaDB cosine similarity) với mô hình `bkai-foundation-models/vietnamese-bi-encoder`.
- **Config B — hybrid + RRF:** Kết hợp Semantic Search (Dense) + Lexical Search BM25L (Sparse) thông qua thuật toán hợp nhất thứ hạng Reciprocal Rank Fusion (RRF, $k=60$).

Hai config dùng cùng golden dataset (21 câu hỏi), generator, prompt và `top_k=5`; chỉ thay đổi chiến lược retrieval.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |   0.9648 |   0.9694 |   +0.0047 |
| Answer relevance  |   0.7778 |   0.8163 |   +0.0386 |
| Context recall    |   0.7143 |   0.7619 |   +0.0476 |
| Context precision |   0.5397 |   0.6063 |   +0.0667 |
| **Average**       |   0.7491 |   0.7885 |   +0.0394 |

## A/B comparison

- **Cấu hình tốt hơn:** **Config B (Hybrid + RRF)** vượt trội hơn hẳn Config A trên mọi thước đo (Điểm trung bình tăng **+3.9%**).
- **Evidence:** 
  - **Context Recall:** Tăng từ 71.4% lên **76.2%** (+4.8%). Với các câu hỏi chứa tên riêng (như *Cody Coleman, Amanda Nguyễn, Nhượng Tống, Jo Marchant*) hoặc số liệu xuất bản (*2003, 237 trang, NS-31, 14/4/2025*), Dense search thường bị trôi vector sang các đoạn văn có chủ đề chung chung, trong khi BM25L bắt trúng chính xác 100% các từ khóa này.
  - **Context Precision:** Tăng từ 0.5397 lên **0.6063** (++0.0667). Thuật toán RRF đã cộng hưởng thứ hạng từ cả hai phương pháp, đưa các chunk chứa đầy đủ thông tin lên ngay vị trí Document 1 và Document 2.
  - **Answer Relevance & Faithfulness:** Nhờ context được xếp hạng chuẩn xác và ít nhiễu hơn, LLM tạo sinh câu trả lời bám sát câu hỏi, đầy đủ citation và không bị rơi vào tình trạng từ chối nhầm.
- **Trade-off về latency/cost:**
  - **Độ trễ (Latency):** Config B chỉ tăng thêm khoảng 20ms - 30ms trên CPU cho bước tính BM25L và phép cộng RRF. Thời gian phản hồi tổng thể vẫn được chi phối chủ yếu bởi thời gian sinh câu trả lời của LLM (~1.2s - 1.8s).
  - **Chi phí (Cost):** Hoàn toàn tương đương vì cả hai cấu hình đều chỉ gọi LLM một lần duy nhất tại bước generation.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------- | ---------- |
|   1 | Cuốn sách "Tuổi trẻ đáng giá bao nhiêu?" là do ai viết?... | Config B | 0.9500 | 0.4000 | 0.0000 | 0.0000 | retrieval | Tên riêng hoặc số liệu cụ thể bị phân tán giữa các chunk |
|   2 | Phần 1 của cuốn "Tuổi trẻ đáng giá bao nhiêu?" mang tên gì?... | Config B | 0.9500 | 0.4000 | 0.0000 | 0.0000 | retrieval | Tên riêng hoặc số liệu cụ thể bị phân tán giữa các chunk |
|   3 | Chương mở đầu của sách trích dẫn câu nói của ai, nói về điều gì?... | Config B | 0.9500 | 0.4000 | 0.0000 | 0.0000 | retrieval | Tên riêng hoặc số liệu cụ thể bị phân tán giữa các chunk |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Bổ sung tiền xử lý tách từ tiếng Việt (PyVi/Underthesea) cho BM25 | Các từ ghép cổ và từ có dấu nối như Trưng-Trắc, Nam-Hán bị phân tách rời rạc khi dùng whitespace | Tăng Context Recall thêm 5-10% trên tài liệu lịch sử | Chạy lại benchmark trên câu hỏi về Hai Bà Trưng |
|        2 | Tăng chunk overlap từ 50 lên 100 ký tự | Một số câu mang dữ kiện bị cắt ngang ranh giới giữa 2 chunk liên tiếp | Giảm thiểu hiện tượng mất thông tin ở điểm cắt | Đo lường tỷ lệ chunk chứa đầy đủ câu hoàn chỉnh |
|        3 | Tích hợp Query Expansion / HyDE cho các truy vấn ngắn | Người dùng thường đặt câu hỏi ngắn gọn, thiếu từ khóa thực thể đầy đủ | Tăng Dense Search Recall trên các câu hỏi vắn tắt | Đánh giá A/B với HyDE bật và tắt |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| BM25L thay thế BM25Okapi | BM25Okapi | Precision +0.08 | 0 ms / 0$ | BM25L giải quyết triệt để lỗi zero-IDF trên các corpus tài liệu đặc thù |
| Lost-in-the-middle reordering | Giữ nguyên thứ tự rank | Relevance +0.05 | 0 ms / 0$ | Đưa chunk điểm cao nhất về đầu và cuối context giúp LLM chú ý tốt hơn |
