Tài liệu này trình bày các phương pháp huấn luyện (training) tối ưu và kiến trúc tiên tiến nhất hiện nay (SOTA) trong việc phân tích tâm trạng đa phương thức (Multimodal Sentiment Analysis - MSA) và các hướng phát triển tương lai.
1. Các phương pháp và kiến trúc train mô hình hiệu quả nhất hiện nay
Chiến lược Hợp nhất Sớm (Early Fusion) kết hợp Transformer: Thay vì xử lý rời rạc, các mô hình SOTA hiện nay sử dụng cơ chế hợp nhất sớm ở cấp độ nhúng (embedding)
. Cụ thể, hệ thống dùng DistilBERT/RoBERTa để mã hóa văn bản, mạng CNN (như MTCNN, ResNet) hoặc ViT cho hình ảnh khuôn mặt, và openSMILE để trích xuất đặc trưng âm thanh (như MFCC) 
. Sau đó, các vector này được ghép nối (concatenate) và đưa qua một mạng Transformer phân loại chung, đạt độ chính xác lên tới 97,87% trên tập dữ liệu CMU-MOSEI 
.
Ứng dụng Mạng Nguyên mẫu và Hàm mất mát phi tương thích (Incongruity Loss): Trong các bài toán phức tạp như phát hiện châm biếm, có sự mâu thuẫn lớn giữa ngữ nghĩa bề mặt của văn bản và biểu cảm/âm thanh thực tế
. Phương pháp tiên tiến nhất là sử dụng mạng nguyên mẫu (prototype-based networks) kết hợp với hàm mất mát phi tương thích (incongruity loss) để chủ động tính toán sự sai lệch giữa cảm xúc tường minh và hàm ý ẩn dụ, qua đó giúp mô hình tự giải thích được quyết định của mình  
.
Huấn luyện nhận biết nhiễu và Đa nhiệm (NAT-MTT): Dữ liệu thực tế thường chứa rất nhiều nhiễu. Phương pháp NAT-MTT (Noise-Aware Multi-Task Transformers) giải quyết vấn đề này bằng cách huấn luyện đa nhiệm (kết hợp trích xuất khía cạnh và phân loại cảm xúc) cùng với một lịch trình tiêm nhiễu nhân tạo (như lỗi chính tả, từ lóng, bỏ sót từ ngẫu nhiên) trực tiếp vào các batch huấn luyện   
. Cách train này giúp giảm tới 42% sự suy giảm hiệu năng do nhiễu và tăng cường độ ổn định khi chuyển giao chéo miền (cross-domain)  
.
Phân tích Kết hợp Tổ hợp (Combinatorial Fusion Analysis - CFA): Thay vì sử dụng một mô hình khổng lồ, CFA kết hợp các mô hình có "độ đa dạng nhận thức" cao (ví dụ: RoBERTa kết hợp cùng các mô hình học máy kinh điển như SVM, Random Forest, XGBoost)  
. Bằng cách áp dụng thuật toán kết hợp điểm số dựa trên trọng số đa dạng (WCDS-SC), hệ thống có thể giảm gần một nửa tỷ lệ lỗi và đạt độ chính xác 97,072% so với việc chỉ dùng mô hình học sâu đơn lẻ 
.
2. Lợi thế của Mô hình Ngôn ngữ Nhỏ (SLMs) tinh chỉnh miền
Thực tế chứng minh rằng, đối với các bài toán phân tích tâm trạng đa phương thức phức tạp hoặc dữ liệu trộn mã (code-mixed), các Mô hình Ngôn ngữ Nhỏ (SLMs) được tinh chỉnh chuyên sâu (như DistilBERT, XLM-RoBERTa, GigaBERT) lại liên tục đánh bại các Mô hình Ngôn ngữ Lớn (LLMs) tổng quát như GPT-4, Llama 3, hay Mistral  
.
Việc sử dụng mô hình cổ điển nhỏ gọn kết hợp tinh chỉnh trên miền dữ liệu cụ thể giúp hệ thống không chỉ chính xác hơn (cải thiện hơn 20% F1-score trong nhiều trường hợp) mà còn giảm thiểu đáng kể chi phí tính toán, rất phù hợp để triển khai thực tế 
.
3. Định hướng phát triển tương lai
Xây dựng AI có khả năng giải thích (Explainable AI - XAI): Xu hướng thiết yếu là phát triển các kỹ thuật giúp con người hiểu rõ cơ chế quyết định của mô hình (ví dụ: tính toán điểm số chú ý để biết hệ thống đang tập trung vào nét mặt hay từ ngữ nào)
.
Tối ưu hóa chạy trên thiết bị đầu cuối theo thời gian thực: Phát triển các mô hình đa phương thức nhẹ và hiệu quả cao để có thể tích hợp trực tiếp lên các thiết bị Edge (như điện thoại, camera an ninh) phục vụ phân tích mạng xã hội, tương tác người-máy và y tế từ xa
.
Kháng nhiễu và mở rộng đa văn hóa: Nâng cao khả năng xử lý dữ liệu bị thiếu hụt, dữ liệu nhiễu và thích ứng với các chuẩn mực văn hóa, ngôn ngữ khác nhau trong giao tiếp con người 
.