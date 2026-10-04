# Phân tích lỗi - Lab 18: Production RAG

**Học viên:** Nguyễn Văn Duy
**Khóa:** K4 - Track 3B
**Lượt đo:** 05/10/2026, Gemini `gemini-3.5-flash-lite`, RAGAS 0.1.x

## Kết quả tổng hợp

RAGAS (khung đánh giá RAG): chấm riêng câu trả lời và ngữ cảnh bằng bốn chỉ số từ 0 đến 1. Điểm cao hơn thể hiện kết quả tốt hơn. Ví dụ Context Recall = 92,5% nghĩa là tập ngữ cảnh bao phủ phần lớn dữ kiện cần dùng; phần còn thiếu là tín hiệu để sửa retrieval hoặc metadata.

| Chỉ số | Naive Baseline | Production | Chênh lệch |
|---|---:|---:|---:|
| Faithfulness (độ trung thực với tài liệu) | 0,9357 | **0,9661** | +0,0304 |
| Answer Relevancy (mức trả lời đúng trọng tâm) | 0,6177 | **0,7834** | +0,1658 |
| Context Precision (độ sạch của ngữ cảnh lấy ra) | 0,8500 | **0,8750** | +0,0250 |
| Context Recall (độ bao phủ dữ kiện cần thiết) | 0,8250 | **0,9250** | +0,1000 |

Production đạt `scores_valid=true`, đủ 20 câu và không thiếu metric. Năm trường hợp dưới đây được lấy trực tiếp từ `reports/ragas_report.json`, sắp theo điểm trung bình tăng dần.

## Bottom-5 Failures

### 1. Câu hỏi nhiều nguồn: phép năm và lương Senior

- **Câu hỏi:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Đáp án chuẩn:** 18 ngày phép; lương Senior P3-P4 là 20-35 triệu đồng/tháng.
- **Câu trả lời thực tế:** Tính đúng 18 ngày phép nhưng nói tài liệu không có mức lương.
- **Chỉ số thấp nhất:** `context_precision = 0,0000`; `context_recall = 0,5000`.
- **Cây lỗi:** Câu trả lời thiếu một nửa dữ kiện -> context có chính sách phép nhưng thiếu bảng lương -> retrieval chỉ lấy hai phiên bản phép năm -> lỗi ở retrieval/reranking theo nhiều chủ đề.
- **Nguyên nhân gốc:** Query gồm hai ý độc lập. Candidate pool bị các đoạn cùng chủ đề phép năm chiếm chỗ, nên không còn `bang_luong_2024.md` trong top-3.
- **Cách sửa:** Tách query thành hai nhánh `phép năm` và `lương Senior`, tìm riêng rồi hợp nhất; thêm diversity theo `source`; chặn nhiều child cùng một parent chiếm toàn bộ top-k trước rerank.

### 2. Câu hỏi số học: phí phạt tạm ứng

- **Câu hỏi:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Đáp án chuẩn:** Quá hạn 5 ngày; 2%/tháng trên 15 triệu là 300.000 đồng/tháng, tính pro-rata khoảng 50.000 đồng cho 5 ngày.
- **Câu trả lời thực tế:** Nêu đúng hạn 15 ngày và mức 2%/tháng nhưng từ chối tính số tiền cụ thể.
- **Chỉ số thấp nhất:** `answer_relevancy = 0,0000`; điểm trung bình 0,5893.
- **Cây lỗi:** Context đúng -> câu hỏi rõ -> model không thực hiện phép tính -> lỗi ở bước synthesis.
- **Nguyên nhân gốc:** System prompt ưu tiên không suy diễn nên model thận trọng quá mức, dù dữ kiện đã đủ cho phép tính `15.000.000 x 2% x 5/30 = 50.000`.
- **Cách sửa:** Bổ sung rule cho câu hỏi số học: chỉ tính khi context chứa đủ số tiền, tỷ lệ, thời gian và đơn vị; buộc trình bày công thức cùng giả định 30 ngày/tháng.

### 3. Xung đột phiên bản: số ngày phép năm

- **Câu hỏi:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Đáp án chuẩn:** Chính sách 2024 cho 15 ngày; bản 2023 có 12 ngày và đã bị thay thế.
- **Câu trả lời thực tế:** Trả lời đúng 15 ngày và nêu chính sách 2024 thay bản 2023.
- **Chỉ số thấp nhất:** `context_precision = 0,5000`.
- **Cây lỗi:** Output đúng -> context chứa cả bản cũ và mới -> bản cũ đứng trước bản hiện hành -> lỗi ở metadata/version reranking.
- **Nguyên nhân gốc:** Cross-encoder đánh relevance theo nội dung nhưng không có rule cứng ưu tiên `effective_date` và trạng thái hiện hành.
- **Cách sửa:** Chuẩn hóa `version`, `effective_date`, `superseded_by`; lọc hoặc giảm điểm tài liệu đã thay thế trước reranking; chỉ giữ bản cũ khi câu hỏi yêu cầu so sánh lịch sử.

### 4. Context Recall: yêu cầu MFA

- **Câu hỏi:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Đáp án chuẩn:** Có; v2.0 bắt buộc MFA, còn v1.0 không yêu cầu.
- **Câu trả lời thực tế:** Trả lời đúng yêu cầu hiện hành và liệt kê phương thức MFA, nhưng không nói rõ bản cũ không yêu cầu.
- **Chỉ số thấp nhất:** `context_recall = 0,5000`.
- **Cây lỗi:** Output đúng phần hiện hành -> đáp án chuẩn yêu cầu cả đối chiếu bản cũ -> context có bản cũ nhưng synthesis bỏ chi tiết -> lỗi ở coverage của câu trả lời.
- **Nguyên nhân gốc:** Prompt tập trung vào quy định đang áp dụng, chưa yêu cầu nêu thay đổi khi context có metadata `ĐÃ THAY THẾ`.
- **Cách sửa:** Khi phát hiện hai phiên bản, tạo một câu đối chiếu ngắn: quy định hiện hành và điểm đã thay đổi; không liệt kê chi tiết ngoài câu hỏi nếu làm loãng trọng tâm.

### 5. Xung đột phiên bản: độ dài mật khẩu

- **Câu hỏi:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Đáp án chuẩn:** v2.0 yêu cầu 12 ký tự; v1.0 yêu cầu 8 ký tự nhưng đã bị thay thế.
- **Câu trả lời thực tế:** Trả lời đúng 12 ký tự và nêu v2.0 thay bản cũ.
- **Chỉ số thấp nhất:** `context_precision = 0,5000`.
- **Cây lỗi:** Output đúng -> context xếp bản v1.0 trước v2.0 -> nhiễu thứ tự bằng chứng -> lỗi ở ranking theo phiên bản.
- **Nguyên nhân gốc:** Reranker nhận diện cả hai văn bản đều liên quan nhưng không dùng tính hiệu lực làm tie-breaker.
- **Cách sửa:** Thêm `is_current` vào metadata và áp dụng tie-breaker tất định sau Cross-Encoder: bản hiện hành trước, bản đã thay thế sau.

## Case study dùng khi trình bày

**Trường hợp chọn:** Nhân viên Senior có 9 năm thâm niên.

1. Câu trả lời đầy đủ không? - Không, thiếu khoảng lương 20-35 triệu đồng/tháng.
2. Context có dữ kiện lương không? - Không; top-3 đều bị nhóm tài liệu phép năm chi phối.
3. Query có hai ý không? - Có: quyền nghỉ phép và khung lương.
4. Sửa ở đâu? - Tách query thành hai retrieval plan, áp dụng diversity theo nguồn, rồi hợp nhất context trước khi sinh câu trả lời.

Nếu có thêm một giờ, ưu tiên giảm latency reranker. Median thực đo là 14.509,76 ms/câu trên CPU, cao hơn nhiều mục tiêu 150 ms của đề. Hướng thử nghiệm là ONNX/FlashRank, giảm candidate từ 20 xuống 10 sau một vòng metadata filter, và cache điểm cho các cặp query-document lặp lại.
