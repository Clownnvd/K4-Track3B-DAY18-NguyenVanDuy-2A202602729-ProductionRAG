# Bài thu hoạch cá nhân - Lab 18: Production RAG

**Họ và tên:** Nguyễn Văn Duy
**MSSV:** 2A202602729
**Khóa:** K4 - Track 3B
**Ngày hoàn thành:** 05/10/2026

## Phần 1: Mapping bài giảng vào code

| Khái niệm | Module và hàm | Quan sát thực tế | Tác dụng đối với quyết định |
|---|---|---|---|
| Semantic Chunking (chia đoạn theo ngữ nghĩa): ghép các câu gần nghĩa dựa trên cosine similarity, trong đó 1 là cùng hướng và 0 là không liên quan | M1 - `chunk_semantic()` | Threshold 0,85 tạo 208 chunk, trung bình 99 ký tự; basic tạo 51 chunk, trung bình 410 ký tự | Chunk ngắn giúp bám câu hỏi hẹp nhưng 208 chunk làm tăng số candidate. Với văn bản có heading rõ, structure-aware phù hợp hơn semantic thuần. |
| Hierarchical Chunking (chia đoạn cha-con): tìm bằng child ngắn nhưng trả parent dài hơn làm context | M1 - `chunk_hierarchical()` | Lượt production tạo 101 child và 26 parent từ 26 tài liệu; `parent_id` không trùng giữa tài liệu | Child tăng độ khớp retrieval; parent giữ đủ điều kiện và ngoại lệ khi sinh câu trả lời. |
| Hybrid Search (tìm kiếm lai): BM25 bắt từ khóa chính xác, Dense Search bắt ý nghĩa; RRF cộng `1/(k + rank + 1)` theo thứ hạng | M2 - `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | Context Recall tăng từ 0,8250 lên 0,9250 | Dùng hybrid khi câu hỏi có cả tên chính sách, con số và cách diễn đạt tự nhiên; không để vector search quyết định hard facts một mình. |
| Cross-Encoder Reranking (xếp hạng lại bằng model đọc đồng thời query và document) | M3 - `CrossEncoderReranker.rerank()` | Median rerank 14.509,76 ms/câu trên CPU; top-3 thường đưa đúng file lên đầu nhưng còn lỗi phiên bản | Chất lượng tăng nhưng latency quá cao. Production cần metadata filter trước, candidate nhỏ hơn hoặc ONNX/FlashRank. |
| RAGAS (khung đánh giá RAG): Faithfulness đo nội dung có được context đỡ; Answer Relevancy đo câu trả lời có đúng trọng tâm; Context Precision đo đoạn đúng có đứng trước; Context Recall đo đủ bằng chứng | M4 - `evaluate_ragas()` và `failure_analysis()` | Production lần lượt đạt 0,9661; 0,7834; 0,8750; 0,9250. Cả bốn cao hơn baseline | Metric chỉ ra tầng cần sửa: Context Precision thấp ở câu xung đột phiên bản; Answer Relevancy thấp ở câu cần tính toán. |
| Enrichment (làm giàu văn bản): thêm context, summary, câu hỏi giả định và metadata trước khi index | M5 - `_enrich_single_call()` | 101/101 child dùng `gemini_combined`; cache giúp lượt production không gọi lại 101 request | Contextual prepend và metadata giúp query tự nhiên khớp chính sách tốt hơn, nhưng dữ kiện do AI sinh không được thay nguồn gốc. |

## Phần 2: Khó khăn và cách giải quyết

### 1. Xung đột Transformers trên Windows

- **Lỗi:** `RuntimeError: PytorchStreamReader failed reading file ... invalid header or archive is corrupted` khi nạp BGE-M3; môi trường còn lấy `transformers 5.1.0` từ Python global.
- **Cách debug:** In phiên bản runtime từ đúng `.venv311`, pin `transformers==4.57.6` và `huggingface-hub==0.36.2` trong môi trường lab. Tải lại trọng số vào ổ D, tính SHA-256 và chỉ dùng file khớp hash chính thức `b5e0ce...aad38`.
- **Bài học:** Test có mock model chưa đủ chứng minh runtime. Cần một smoke test nạp model và encode thật.

### 2. Thiếu dung lượng ổ C

- BGE-M3 có file trọng số khoảng 2,27 GB; cache mặc định trên ổ C không đủ chỗ.
- Chuyển cache của riêng lab sang `D:/Codex-Lab18-2026-10-04-Models/hub` qua `BGE_CACHE_DIR`. `.env` bị loại khỏi Git nên đường dẫn và key không đi vào repo.

### 3. Gemini free tier giới hạn 15 request/phút

- **Lỗi:** `429 RESOURCE_EXHAUSTED ... limit: 15, model: gemini-3.5-flash-lite` trong lượt RAGAS đầu tiên.
- **Cách sửa:** Serialize các request chưa cache, chờ tối thiểu 4,2 giây và retry có khoảng nghỉ khi gặp 429. Giảm `answer_relevancy.strictness` từ 3 xuống 1; metric vẫn là RAGAS Answer Relevancy nhưng chỉ sinh một câu hỏi phụ cho mỗi mẫu.
- **Kết quả:** Hai báo cáo đều đủ 20 câu, `status=complete`, `scores_valid=true`, không có metric NaN.

### 4. Bài toán phiên bản và câu hỏi nhiều ý

Reranker đánh giá mức liên quan nội dung tốt nhưng không hiểu chắc chắn tài liệu nào đã hết hiệu lực. Hai failure rõ nhất đều đưa bản cũ lên trước bản hiện hành. Câu Senior 9 năm còn thiếu tài liệu lương vì ba slot bị tài liệu phép chiếm. Cần kết hợp model score với rule metadata và source diversity.

## Phần 3: Action Plan cho GrantFinder

### Hiện trạng

GrantFinder đang dùng BM25 kết hợp local vector, rule kiểm deadline/eligibility và reranker cơ bản. Dữ liệu Grants.gov và NAFOSTED đã có canonical URL, source ID và source span. Các điểm yếu còn lại là tài liệu dài nhiều phụ lục, phiên bản form thay đổi, retrieval latency và chưa có benchmark theo từng loại câu hỏi.

### Kế hoạch áp dụng

1. **Chunking:** Dùng structure-aware cho thông báo và biểu mẫu có heading; dùng hierarchical parent-child cho PDF dài. Giữ `source_span`, call ID, package ID và form version trong mọi child.
2. **Search:** Hybrid BM25 + Dense + RRF. Lọc cứng theo nguồn quỹ, trạng thái, deadline và loại applicant trước khi lấy top-k.
3. **Reranking:** Cross-encoder chỉ chạy sau metadata filter. Benchmark BGE reranker với ONNX/FlashRank; mục tiêu median dưới 1 giây trên máy demo thay vì 14,5 giây.
4. **Evaluation:** Dùng bốn metric RAGAS cho phần trả lời, thêm metric tất định cho deadline, tiền tệ, eligibility và trích dẫn. Gold set phải có câu xung đột phiên bản và câu nhiều call.
5. **Enrichment:** Dùng contextual prepend và HyQA cho đoạn thiếu tiêu đề; metadata do parser/rule tạo giữ quyền quyết định. AI summary chỉ hỗ trợ retrieval, không thay hard facts.

### Timeline

- **Tuần 1:** Chuẩn hóa metadata version/status, thêm source diversity và query decomposition.
- **Tuần 2:** Benchmark ba cấu hình reranker trên cùng gold set; chọn cấu hình đạt chất lượng nhưng đủ nhanh cho demo.
- **Tuần 3:** Chạy RAGAS cùng metric hard-fact, phân tích bottom-10 và đưa regression cases vào CI.
