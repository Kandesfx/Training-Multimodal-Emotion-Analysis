PHÂN TÍCH KỸ THUẬT VÀ HƯỚNG PHÁT TRIỂN HỆ THỐNG AI ĐA PHƯƠNG THỨC TIÊN TIẾN
1. Tổng quan về xu hướng và thách thức của Mô hình Đa phương thức (Multimodal AI)
Trong bối cảnh trí tuệ nhân tạo hiện đại, chúng ta đang chứng kiến một cuộc hội tụ chiến lược giữa Thị giác máy tính (Vision) và Xử lý ngôn ngữ tự nhiên (NLP). Sự dịch chuyển từ các mô hình đơn lẻ (unimodal) sang các hệ thống đa phương thức (multimodal) không đơn thuần là việc kết hợp dữ liệu, mà là hướng tới khả năng thấu hiểu ngữ cảnh sâu (deep contextual understanding). Điều này đặc biệt quan trọng trong các tác vụ yêu cầu khả năng suy luận phức tạp như phát hiện châm biếm (sarcasm detection) – nơi tồn tại sự mâu thuẫn giữa ý nghĩa hiển ngôn và hàm ẩn, hoặc trả lời câu hỏi dựa trên văn bản trong ảnh (TextVQA), vốn đòi hỏi sự căn chỉnh chính xác giữa các thực thể thị giác và thực thể ngữ nghĩa.
Tuy nhiên, việc triển khai các hệ thống này vẫn đối mặt với hai rào cản mang tính hệ thống. Thứ nhất là tính chất "hộp đen" (black-box) của Deep Learning, gây khó khăn cho việc giải thích các quyết định của mô hình trong các bài toán nhạy cảm. Thứ hai là sự thiếu hụt dữ liệu chất lượng cao cho các ngôn ngữ ít nguồn lực (low-resource) như tiếng Việt. Để giải quyết, chúng ta cần những kiến trúc không chỉ mạnh về hiệu suất mà còn phải bền vững trước nhiễu và có tính diễn giải cao.
2. Kiến trúc Đa tầng: Sự kết hợp giữa Semantic View và Sentiment View
Dựa trên các nghiên cứu mới nhất về phát hiện châm biếm, kiến trúc hệ thống nên được phân tách thành hai luồng xử lý độc lập để khai thác tối đa đặc trưng đa chiều:
So sánh các thành phần kiến trúc cốt lõi
Tiêu chí
Semantic View (Luồng Ngữ nghĩa)
Sentiment View (Luồng Cảm xúc)
Mục tiêu trích xuất
Nắm bắt ý nghĩa ngữ cảnh toàn cục (contextual meaning).
Phân tích sự tương phản giữa cảm xúc hiển ngôn và hàm ẩn.
Encoder sử dụng
RoBERTa-large hoặc Sentence-BERT (all-mpnet-base-v2).
SiEBERT (Fine-tuned RoBERTa cho Sentiment).
Embedding đầu ra
Sentence-level Contextual Embedding.
Explicit Sentiment Vector & Implicit Sentiment Vector.
Cơ chế Incongruity Loss (Mất mát do sự không nhất quán)
Cơ chế này sử dụng hàm Cross-Entropy để đo lường sự khác biệt giữa cảm xúc hiển ngôn (trích xuất từ các từ ngữ mang sắc thái trực tiếp) và cảm xúc hàm ẩn (suy luận từ ngữ cảnh). Bằng cách sử dụng SiEBERT làm nhãn tham chiếu cho các thành phần cảm xúc, mô hình học cách nhận diện sự mâu thuẫn – chìa khóa cốt lõi để giải mã ý nghĩa châm biếm.
3. Tối ưu hóa căn chỉnh phương thức bằng SIGROT và Unbalanced Optimal Transport
Kỹ thuật SIGROT (Similarity-Graph Regularized Optimal Transport) cung cấp một phương pháp căn chỉnh vượt trội so với Contrastive Learning truyền thống (vốn chỉ tập trung vào các cặp thực thể đơn lẻ).
Kiến trúc Backbone: Sử dụng DINOv3-based Vision Transformer cho luồng hình ảnh và Vietnamese Sentence-BERT cho luồng văn bản.
Quy trình logic của SIGROT:
Xây dựng Similarity Graph: Tính toán cấu trúc quan hệ nội tại (intra-modality) trong mỗi batch.
Thiết lập Transport Cost: Xác định khoảng cách embedding giữa ảnh và văn bản.
Tối ưu hóa Unbalanced Optimal Transport (UOT): Việc sử dụng biến thể "Unbalanced" cho phép mô hình xử lý hiệu quả các dữ liệu nhiễu hoặc không nhất quán giữa hai phương thức, tránh việc ép buộc căn chỉnh sai lệch.
Căn chỉnh cấu trúc toàn cầu: Thay vì chỉ so khớp cặp (pairwise), SIGROT bảo toàn cấu trúc đồ thị tương đồng, giúp giảm thiểu Modality Gap hiệu quả trong điều kiện ít dữ liệu.
4. Chiến lược đào tạo chống nhiễu (NAT) và Tính diễn giải (Interpretability)
Dữ liệu thực tế luôn tồn tại sai số. Do đó, tôi đề xuất áp dụng khung Noise-Aware Training (NAT) kết hợp với suy luận dựa trên nguyên mẫu (Prototype-based Reasoning).
Cấu hình chiến lược NAT (Noise-Aware Training)
Hệ thống sẽ tiêm nhiễu có kiểm soát (Synthetic Noise) bao gồm: Sai lỗi chính tả, Word Dropout, Thay thế đồng nghĩa và Tiếng lóng.
Curriculum Schedule: Áp dụng lộ trình tăng dần với thời gian khởi động T 
ramp
​
 =5000 bước.
Noise Ratio: Duy trì tỷ lệ nhiễu β=0.5 trong mỗi batch để đảm bảo mô hình học được các đặc trưng bất biến (noise-invariant).
Prototype-based Reasoning và Tính diễn giải
Thay vì các trọng số Attention khó hiểu, chúng ta triển khai Prototype Layer:
Khởi tạo: Sử dụng thuật toán k-means clustering trên tập huấn luyện để xác định các tâm cụm (cluster centers), thiết lập 20 nguyên mẫu (prototypes) cho mỗi lớp.
Prototype Projection: Chiếu các vector nguyên mẫu lên các điểm dữ liệu thực tế gần nhất trong tập huấn luyện. Điều này cho phép mô hình đưa ra giải thích: "Mẫu dữ liệu này được phân loại là X vì nó có cấu trúc tương đồng với ví dụ thực tế Y".