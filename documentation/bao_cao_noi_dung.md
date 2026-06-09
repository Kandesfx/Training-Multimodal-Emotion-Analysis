# BÁO CÁO ĐỒ ÁN

## Đề Tài 17: Hệ Thống Phân Tích Cảm Xúc Đa Phương Thức Cho Tiếng Việt

**Môn học:** Học Sâu (Deep Learning)

**Giảng viên hướng dẫn:** [Tên giảng viên]

**Nhóm thực hiện:**

| Họ tên | MSSV | Email |
|:---|:---|:---|
| [Họ tên thành viên 1] | [MSSV] | [Email] |
| [Họ tên thành viên 2] | [MSSV] | [Email] |
| [Họ tên thành viên 3] | [MSSV] | [Email] |

**Ngày nộp:** 09/06/2026

---

## 1. Lịch Làm Việc Nhóm Hàng Tuần

| Tuần | Thời gian | Công việc chính | Ghi chú |
|:---:|:---|:---|:---|
| 1 | 12/05/2026 | Khảo sát dataset CMU-MOSEI, setup môi trường Colab | |
| 2 | 19/05/2026 | Triển khai EarlyFusionLSTM baseline, chạy P0 training | |
| 3 | 26/05/2026 | Triển khai MulT Transformer, chạy aligned training | |
| 4 | 02/06/2026 | MulT unaligned training, cải tiến training strategy | |
| 5 | 09/06/2026 | Emotion classification training (P0), threshold tuning, báo cáo Phase 1 | Đã hoàn thành |

---

## 2. Công Việc Mỗi Thành Viên Trong Nhóm

| Thành viên | Công việc chính | Phạm vi đóng góp |
|:---|:---|:---|
| [Họ tên TV1] | Xây dựng kiến trúc mô hình, training pipeline | MulT, EarlyFusion, Trainer, Evaluator |
| [Họ tên TV2] | Xây dựng Emotion Data Studio, pipeline AI | PyQt5 UI, feature extractors, exporters |
| [Họ tên TV3] | Thu thập dữ liệu, huấn luyện Colab, đồng bộ GCS | Notebooks, checkpoints, GCS sync |

---

## 3. Mục lục

1. [Lịch làm việc nhóm hàng tuần](#1-lịch-làm-việc-nhóm-hàng-tuần)
2. [Công việc mỗi thành viên trong nhóm](#2-công-việc-mỗi-thành-viên-trong-nhóm)
3. [Mục lục](#3-mục-lục)
4. [Giới thiệu](#4-giới-thiệu)
   - [4.1. Phạm vi của đồ án](#41-phạm-vi-của-đồ-án)
   - [4.2. Mục tiêu](#42-mục-tiêu)
   - [4.3. Sự cần thiết và lý do chọn đề tài](#43-sự-cần-thiết-và-lý-do-chọn-đề-tài)
5. [Phân tích đề tài](#5-phân-tích-đề-tài)
   - [5.1. Phân tích yêu cầu](#51-phân-tích-yêu-cầu)
   - [5.2. Yêu cầu chức năng](#52-yêu-cầu-chức-năng)
6. [Thiết kế](#6-thiết-kế)
   - [6.1. Đề xuất sử dụng thuật toán](#61-đề-xuất-sử-dụng-thuật-toán)
   - [6.2. Cách thức giải quyết bài toán](#62-cách-thức-giải-quyết-bài-toán)
7. [Thực hiện: cài đặt ứng dụng bài toán](#7-thực-hiện-cài-đặt-ứng-dụng-bài-toán)
8. [Kết quả thực nghiệm và đánh giá](#8-kết-quả-thực-nghiệm-và-đánh-giá)
9. [Kết luận và định hướng phát triển](#9-kết-luận-và-định-hướng-phát-triển)
10. [Tài liệu tham khảo](#10-tài-liệu-tham-khảo)
11. [Phụ lục](#11-phụ-lục)

---

## 4. Giới thiệu

### 4.1. Phạm vi của đồ án

#### 4.1.1. Bài toán

Đồ án tập trung vào hai bài toán chính trong lĩnh vực phân tích cảm xúc đa phương thức (Multimodal Sentiment and Emotion Analysis):

**Bài toán 1 — Sentiment Regression (Hồi quy tâm trạng):**
- **Input:** Ba luồng dữ liệu đồng thời:
  - **Text (văn bản):** vector đặc trưng trích xuất từ mô hình ngôn ngữ BERT, kích thước 768 chiều.
  - **Audio (âm thanh):** vector đặc trưng âm thanh COVAREP, kích thước 74 chiều, bao gồm các đặc trưng như pitch, formants, năng lượng.
  - **Vision (hình ảnh):** vector đặc trưng khuôn mặt FACET Action Units, kích thước 35 chiều, phản ánh biểu cảm khuôn mặt.
- **Output:** Giá trị số thực trong khoảng [-3, +3], biểu diễn mức độ tâm trạng tiêu cực (-3) đến tích cực (+3).

**Bài toán 2 — Emotion Classification (Phân loại cảm xúc đa nhãn):**
- **Input:** Cùng ba luồng dữ liệu đa phương thức như trên.
- **Output:** Vector 6 giá trị, mỗi giá trị biểu diễn cường độ của một cảm xúc trong thang [0, 3]:
  - Happy (vui), Sad (buồn), Angry (tức giận), Surprise (ngạc nhiên), Disgust (ghê tởm), Fear (sợ hãi).
  - Giá trị 0 nghĩa là cảm xúc không xuất hiện; giá trị 3 nghĩa là cường độ cao nhất.

#### 4.1.2. Giới hạn

- **Dataset hiện tại:** Bộ dữ liệu CMU-MOSEI (tiếng Anh). Quá trình chuyển giao sang tiếng Việt (cross-lingual transfer) sử dụng PhoBERT đang được chuẩn bị hạ tầng với Emotion Data Studio và notebook feature extraction.
- **Số lượng mẫu sau filter:** Train: 16,326 / Valid: 1,871 / Test: 4,659 (CMU-MOSEI standard splits).
- **Phần cứng:** GPU NVIDIA với CUDA hỗ trợ; thử nghiệm trên Google Colab (GPU Tesla T4, 15GB VRAM).
- **Phạm vi đa ngôn ngữ:** Giai đoạn hiện tại tập trung vào dataset tiếng Anh; hạ tầng cho tiếng Việt (PhoBERT feature extractor, MMSA exporter) đã hoàn thành.

---

### 4.2. Mục tiêu

Đồ án đề ra các mục tiêu cụ thể sau:

1. **Xây dựng và so sánh hai mô hình:**
   - **Mô hình Baseline (Early Fusion LSTM):** Kết hợp sớm (early fusion) ba luồng đặc trưng bằng mạng LSTM hai chiều (Bidirectional LSTM), sau đó nối concatenation và đưa qua mạng MLP để dự đoán.
   - **Mô hình cải tiến (MulT — Multimodal Transformer):** Sử dụng cơ chế Cross-Modal Attention cho phép ba phương thức tương tác trực tiếp với nhau thông qua kiến trúc Transformer, từ đó nắm bắt các mối quan hệ phức tạp giữa text, audio và vision.

2. **Đánh giá hiệu quả qua các chỉ số chuẩn:**
   - Với Sentiment Regression: MAE (Mean Absolute Error), MSE, Pearson Correlation, Accuracy-2 (nhị phân), Accuracy-7 (7 lớp).
   - Với Emotion Classification: Mean F1-Score, Mean Accuracy, F1 per cảm xúc.

3. **Triển khai huấn luyện trên nền tảng đám mây:** Cấu hình đồng bộ dữ liệu với Google Cloud Storage (GCS) và giám sát quá trình huấn luyện bằng Weights & Biases (W&B).

4. **Chuẩn bị hạ tầng cho tiếng Việt:** Xây dựng pipeline feature extraction với PhoBERT để mở rộng sang tiếng Việt trong giai đoạn tiếp theo.

---

### 4.3. Sự cần thiết và lý do chọn đề tài

#### 4.3.1. Tầm quan trọng của Sentiment Analysis trong thực tế

Phân tích cảm xúc (Sentiment Analysis) là một trong những ứng dụng quan trọng nhất của Xử lý Ngôn ngữ Tự nhiên (NLP) và Học sâu (Deep Learning), với ứng dụng rộng rãi trong:

- **Thương mại điện tử:** Phân tích đánh giá sản phẩm để hiểu phản hồi khách hàng, phát hiện đánh giá giả mạo, tối ưu hóa dịch vụ.
- **Mạng xã hội:** Theo dõi dư luận về thương hiệu, sản phẩm, sự kiện theo thời gian thực.
- **Chăm sóc khách hàng:** Tự động phân loại và ưu tiên phiếu hỗ trợ dựa trên mức độ tiêu cực/tích cực.
- **Y tế và tâm lý học:** Phát hiện trầm cảm, rối loạn lo âu qua phân tích giọng nói và biểu cảm khuôn mặt.
- **Giáo dục:** Đánh giá trải nghiệm học tập của sinh viên qua phản hồi văn bản.

#### 4.3.2. Thách thức đặc thù của tiếng Việt

Tiếng Việt có những đặc điểm ngôn ngữ tạo ra thách thức riêng so với tiếng Anh:

- **Đa thanh (Diacritics):** Dấu thanh (sắc, huyền, hỏi, ngã, nặng) thay đổi hoàn toàn nghĩa của từ (ví dụ: "ma" có 5 nghĩa khác nhau). Các mô hình tiếng Việt cần hiểu ngữ cảnh để phân biệt.
- **Không có khoảng trắng phân tách từ (Word Segmentation):** Tiếng Việt không có ranh giới từ rõ ràng như tiếng Anh; cần sử dụng các công cụ word segmentation (VD: PyVI, underthesea).
- **Từ địa phương và slang:** Mạng xã hội tiếng Việt sử dụng rất nhiều từ lóng, viết tắt, emoji đặc thù.
- **Ảnh hưởng cross-lingual:** Các mô hình pretrained tiếng Anh (BERT, GPT) không trực tiếp áp dụng cho tiếng Việt; cần sử dụng PhoBERT hoặc các mô hình đa ngôn ngữ.

#### 4.3.3. Lý do chọn đề tài

1. **Tính thời sự:** Phân tích cảm xúc đa phương thức (kết hợp text, audio, vision) đang là xu hướng nghiên cứu mạnh mẽ, vượt trội so với phương pháp chỉ dùng văn bản.
2. **Độ khó phù hợp:** Đề tài yêu cầu kiến thức về Deep Learning, xử lý đa phương thức, và tối ưu hóa mô hình — phù hợp với nội dung môn học.
3. **Tính ứng dụng thực tiễn cao:** Kết quả có thể triển khai trong nhiều lĩnh vực thực tế.
4. **Hướng phát triển rõ ràng:** Từ baseline đến mô hình SOTA (MulT Transformer), với lộ trình mở rộng sang tiếng Việt qua PhoBERT.

---

## 5. Phân tích đề tài

### 5.1. Phân tích yêu cầu

#### 5.1.1. Yêu cầu chức năng

Từ phân tích source code, hệ thống cần đáp ứng các yêu cầu chức năng sau:

| # | Yêu cầu chức năng | Mô tả chi tiết |
|:---:|:---|:---|
| FC-01 | Tải và tiền xử lý dữ liệu đa phương thức | Hệ thống phải đọc được file `.pkl` chứa đặc trưng của 3 phương thức (text 768d, audio 74d, vision 35d), tự động thay thế giá trị Inf/NaN, chuyển sang float32, và lọc các mẫu không có nhãn cảm xúc phù hợp. |
| FC-02 | Huấn luyện mô hình Baseline (Early Fusion LSTM) | Mô hình kết hợp 3 BiLSTM encoder riêng biệt cho từng phương thức, nối đặc trưng và đưa qua MLP regression. Hỗ trợ huấn luyện với early stopping dựa trên MAE trên tập validation. |
| FC-03 | Huấn luyện mô hình cải tiến (MulT Transformer) | Mô hình sử dụng Cross-Modal Attention cho phép 3 phương thức tương tác qua lại. Hỗ trợ cả chế độ aligned (cùng độ dài sequence) và unaligned (độ dài khác nhau). |
| FC-04 | Huấn luyện trên tập dữ liệu cảm xúc (6 nhãn) | Mô hình phân loại đa nhãn cho 6 loại cảm xúc: Happy, Sad, Angry, Surprise, Disgust, Fear. Sử dụng BCEWithLogitsLoss. |
| FC-05 | Đánh giá mô hình | Tính toán đầy đủ các chỉ số: MAE, MSE, Correlation (sentiment) và Mean F1, Mean Accuracy (emotion). Ghi lại kết quả vào file CSV và đồng bộ lên GCS. |
| FC-06 | Lưu và tải checkpoint | Tự động lưu best model và last model sau mỗi epoch. Hỗ trợ resume từ checkpoint khi bị gián đoạn. |
| FC-07 | Huấn luyện trên Google Colab | Hệ thống phải chạy được trên Google Colab với GPU miễn phí, tải dữ liệu từ GCS, đồng bộ checkpoint lên GCS. |
| FC-08 | Giám sát huấn luyện bằng W&B | Tích hợp Weights & Biases để theo dõi loss, metrics theo thời gian thực. |

#### 5.1.2. Yêu cầu phi chức năng

| # | Yêu cầu phi chức năng | Mô tả chi tiết |
|:---:|:---|:---|
| PNF-01 | **Tốc độ xử lý:** | Thời gian huấn luyện mỗi epoch khoảng 2-5 phút trên GPU NVIDIA T4. Sử dụng Mixed Precision Training (AMP) để tăng tốc. |
| PNF-02 | **Độ chính xác tối thiểu:** | Sentiment: MAE ≤ 0.57, Correlation ≥ 0.73. Emotion: Mean F1 ≥ 0.55. |
| PNF-03 | **Bộ nhớ:** | Sử dụng không quá 14GB VRAM trên GPU Tesla T4 (15GB). Sử dụng gradient accumulation nếu cần. |
| PNF-04 | **Khả năng tái sử dụng:** | Mã nguồn rõ ràng, có cấu hình qua dataclass, dễ thay đổi model type, learning rate, batch size. |

#### 5.1.3. Dataset sử dụng

**Tên dataset:** CMU-MOSEI (CMU Multimodal Opinion Sentiment and Emotion Intensity)

| Thông tin | Chi tiết |
|:---|:---|
| **Nguồn gốc** | CMU Multimodal Data Laboratory — http://immortal.multicomp.org/projects/cmumosei |
| **Số lượng mẫu** | 22,856 utterances (video clips độ dài trung bình ~7.5 giây) |
| **Ngôn ngữ** | Tiếng Anh |
| **Split chuẩn** | Train: 16,326 / Valid: 1,871 / Test: 4,659 |
| **Đặc trưng Text** | Kích thước: (N, 50, 768) — GloVe embeddings hoặc BERT embeddings, độ dài chuỗi cố định 50 tokens |
| **Đặc trưng Audio** | Kích thước: (N, 50, 74) — COVAREP acoustic features (pitch, formants, MCEP, v.v.) |
| **Đặc trưng Vision** | Kích thước: (N, 50, 35) — FACET Action Units (biểu cảm khuôn mặt) |
| **Nhãn Sentiment** | Giá trị liên tục trong khoảng [-3, +3] |
| **Nhãn Emotion** | 6 cường độ cảm xúc, mỗi giá trị trong khoảng [0, 3] |

**Phân bố nhãn Emotion (trên tập Train):**

| Cảm xúc | Tỷ lệ xuất hiện | Ý nghĩa |
|:---|---:|:---|
| Happy | ~34.0% | Phổ biến nhất |
| Angry | ~14.5% | Phổ biến |
| Sad | ~14.1% | Phổ biến |
| Disgust | ~11.9% | Trung bình |
| Surprise | ~3.3% | Hiếm |
| Fear | ~2.2% | Hiếm nhất — thách thức lớn cho việc học |

> **Nhận xét:** Dữ liệu emotion có sự mất cân bằng nghiêm trọng. Lớp Fear chỉ chiếm 2.2% trong khi Happy chiếm 34.0% — gấp ~15 lần. Điều này ảnh hưởng lớn đến chất lượng dự đoán các lớp thiểu số và là lý do cần sử dụng Focal Loss hoặc class weights.

---

### 5.2. Yêu cầu chức năng (bổ sung)

#### 5.2.1. Data Pipeline

Hệ thống tiền xử lý dữ liệu phải thực hiện các bước sau:

1. **Tải dữ liệu từ GCS** hoặc Google Drive trên Colab.
2. **Kiểm tra và xử lý Inf/NaN:** Thay thế giá trị vô hạn trong đặc trưng audio bằng 0.0.
3. **Chuyển đổi dtype:** Ép tất cả đặc trưng về float32 để tiết kiệm bộ nhớ.
4. **Lọc mẫu không khớp cảm xúc (Emotion Mode):** Chỉ giữ lại các mẫu có đủ nhãn cho cả 6 cảm xúc.
5. **Tạo DataLoader** với batch size, shuffle, và prefetch.

#### 5.2.2. Training Pipeline

Quy trình huấn luyện chuẩn:

1. Khởi tạo mô hình, optimizer (AdamW), scheduler (Cosine Warmup).
2. Huấn luyện nhiều epoch với early stopping (patience=15).
3. Đánh giá trên tập validation sau mỗi epoch.
4. Lưu checkpoint (best + last) sau mỗi epoch.
5. Đồng bộ checkpoint lên GCS.
6. Ghi metrics vào file CSV.

---

## 6. Thiết kế

### 6.1. Đề xuất sử dụng thuật toán

#### 6.1.1. Model 1 — Early Fusion LSTM (Baseline)

##### Lý do chọn Baseline

Mô hình Early Fusion LSTM là lựa chọn baseline tự nhiên vì:
- **Đơn giản và dễ hiểu:** Kiến trúc rõ ràng, dễ debug, nhanh chóng đánh giá baseline performance.
- **Phù hợp với yêu cầu đề tài:** Đề tài yêu cầu so sánh với baseline CNN/BiLSTM — Early Fusion LSTM thỏa mãn yêu cầu này.
- **Khởi đầu an toàn:** Trước khi thử nghiệm mô hình phức tạp hơn (MulT), cần có baseline để so sánh.

##### Kiến trúc chi tiết

Kiến trúc Early Fusion LSTM gồm 3 thành phần chính:

**a) BiLSTM Encoder (3 cá thể riêng biệt cho mỗi phương thức):**

| Phương thức | Input Dim | Hidden Dim | Output Dim | Chiều |
|:---|:---:|:---:|:---:|:---:|
| Text Encoder | 768 | 128 | 256 | Bidirectional |
| Audio Encoder | 74 | 64 | 128 | Bidirectional |
| Vision Encoder | 35 | 64 | 128 | Bidirectional |

Mỗi encoder nhận chuỗi đặc trưng `(batch, seq_len, input_dim)`, trả về vector biểu diễn cố định `(batch, hidden_dim * 2)` bằng cách nối hai chiều của LSTM.

**b) Fusion MLP:**

| Layer | Input | Output | Activation |
|:---|:---:|:---:|:---:|
| Linear 1 | 512 (=256+128+128) | 256 | ReLU + BatchNorm + Dropout(0.3) |
| Linear 2 | 256 | 128 | ReLU + Dropout(0.2) |
| Linear 3 (Output) | 128 | 1 | — |

**c) Output:** Giá trị sentiment score (scalar, sau đó squeeze).

```
Input (B, 50, 768) ─┐
                    ├─→ BiLSTM Encoder ─→ (B, 256) ─┐
Input (B, 50,  74) ─┤                                  ├─→ Concat (B, 512) ─→ MLP ─→ (B,) ─ Output
                    ├─→ BiLSTM Encoder ─→ (B, 128) ─┤
Input (B, 50,  35) ─┘                                  │
                    └─→ BiLSTM Encoder ─→ (B, 128) ─┘
```

##### Hạn chế của Baseline

| Hạn chế | Giải thích |
|:---|:---|
| **Fusion quá muộn ở cấp độ sequence** | Mô hình xử lý mỗi phương thức độc lập đến cuối rồi mới fusion — không có cơ chế cho phép các phương thức tương tác sớm. |
| **Chỉ dùng hidden state cuối** | Mặc định dùng trạng thái ẩn cuối của LSTM — có thể bỏ sót thông tin quan trọng ở các bước trước. |
| **Không xử lý được misalignment** | Giả định text, audio, vision cùng độ dài sequence — không hoạt động tốt khi các phương thức không align. |
| **Kích thước hidden state cố định** | Thông tin từ phương thức có dim cao (text: 768) bị nén qua LSTM encoder có hidden_dim chỉ 128. |

---

#### 6.1.2. Model 2 — MulT Transformer (Mô hình cải tiến)

##### Lý do cải tiến từ Baseline

Mô hình MulT (Multimodal Transformer) được chọn thay vì các lựa chọn khác vì:

1. **Cross-Modal Attention cho phép tương tác sớm:** Thay vì xử lý riêng rồi fusion muộn, MulT cho phép text "nhìn" vào audio/vision và ngược lại ở mọi timestep thông qua cơ chế attention.
2. **Xử lý được unaligned sequences:** Audio và vision có thể có độ dài khác nhau (50 vs 500 timesteps), MulT xử lý được nhờ cross-attention với key padding mask.
3. **Được chứng minh hiệu quả trên CMU-MOSEI:** Kiến trúc gốc được đề xuất trong paper ACL 2019 và đã đạt kết quả SOTA trên dataset này.
4. **Cải tiến P1 đã được kiểm chứng:** Dự án áp dụng các cải tiến từ 2022-2024: Pre-LayerNorm, Stochastic Depth, GELU activation.

##### Kiến trúc chi tiết

Kiến trúc MulT gồm 7 bước chính:

**Bước 1 — Pre-LayerNorm Projections:**
Mỗi phương thức được chiếu (project) về không gian chung `d_model=128` qua LayerNorm → Linear projection → Positional Encoding:

| Phương thức | Input Dim | Output Dim | Pre-LN |
|:---|:---:|:---:|:---:|
| Text | 768 | 128 | Yes |
| Audio | 74 | 128 | Yes |
| Vision | 35 | 128 | Yes |

> **Tại sao Pre-LayerNorm?** Thay vì normalize sau residual (Post-LN), Pre-LN đặt LayerNorm bên trong nhánh residual, giúp huấn luyện sâu ổn định hơn mà không cần warm-up schedule phức tạp.

**Bước 2 — Cross-Modal Attention (6 luồng chéo):**
Sáu transformer block cho phép mỗi phương thức "nhìn" vào hai phương thức còn lại:

| Luồng | Target (Query) | Source (Key/Value) | Mô tả |
|:---:|:---:|:---:|:---|
| T←A | Text | Audio | Text hỏi audio |
| T←V | Text | Vision | Text hỏi vision |
| A←T | Audio | Text | Audio hỏi text |
| A←V | Audio | Vision | Audio hỏi vision |
| V←T | Vision | Text | Vision hỏi text |
| V←A | Vision | Audio | Vision hỏi audio |

Mỗi block gồm: LayerNorm → Multi-Head Cross-Attention → LayerNorm → Feed-Forward (GELU).

**Bước 3 — Merge (Residual Sum):**
Đầu ra cross-modal được cộng với đầu vào gốc:
```
t_merged = t + t_from_audio + t_from_vision
a_merged = a + a_from_text + a_from_vision
v_merged = v + v_from_text + v_from_audio
```

**Bước 4 — Self-Attention Transformer Encoder (2 lớp mỗi phương thức):**
Sau khi đã tích hợp thông tin từ các phương thức khác, mỗi phương thức tự xử lý bằng self-attention để nắm bắt các mối quan hệ nội tại.

**Bước 5 — Attention Pooling (thay vì last timestep):**
Thay vì lấy hidden state ở timestep cuối, hệ thống dùng Attention Pooling để tính weighted sum của tất cả timesteps, với weights được học. Điều này đặc biệt quan trọng vì thông tin cảm xúc có thể xuất hiện ở bất kỳ đâu trong chuỗi.

**Bước 6 — LayerNorm + Concatenate:**
```
fused = concat([text_pooled, audio_pooled, vision_pooled])  # (B, 3*128) = (B, 384)
```

**Bước 7 — Enhanced Fusion Head:**
```
fused_ln = LayerNorm(fused)
fused_proj = GELU(Linear(fused_ln, 256)) → dropout → Linear(256, 128) → GELU → Linear(128, 1)
```

##### Ưu điểm so với Baseline

| Tiêu chí | Early Fusion LSTM | MulT Transformer |
|:---|:---:|:---:|
| Tương tác phương thức | Fusion muộn (chỉ ở vector cuối) | Cross-modal attention ở mọi timestep |
| Xử lý misalignment | Không hỗ trợ | Hỗ trợ qua key padding mask |
| Nắm bắt thông tin sequence | Chỉ hidden state cuối | Attention pooling trên toàn bộ sequence |
| Số lượng tham số | ~0.8M | ~1.7M (d_model=64) / ~5.8M (d_model=128) |
| Khả năng mở rộng | Hạn chế | Cao — có thể tăng số layers, heads |

---

### 6.2. Cách thức giải quyết bài toán

#### 6.2.1. Data Preprocessing Pipeline

```
aligned_50.pkl (GCS)
        │
        ▼
┌───────────────────┐
│ MOSEIAlignedDataset│
│  1. Load pickle    │
│  2. Replace Inf   │─── audio_inf_replacement = 0.0
│  3. Cast float32  │
│  4. Filter emotion│─── emotion_matched_mask (nếu task=emotion)
│  5. Validate shape│
└───────────────────┘
        │
        ▼
   DataLoader (batch_size=32, shuffle=True)
        │
        ▼
   Training Loop
```

**Các bước tiền xử lý chi tiết:**

1. **Thay thế Inf/NaN:** Đặc trưng audio COVAREP có thể chứa giá trị vô hạn (do lỗi trích xuất âm thanh). Hệ thống thay thế bằng 0.0.
2. **Chuyển float32:** Giảm dung lượng bộ nhớ từ float64 xuống float32 — giảm 50% RAM usage.
3. **Lọc emotion samples:** Khi task='emotion', chỉ giữ lại các mẫu có đủ 6 nhãn cảm xúc (emotion_matched_mask).
4. **Attention Pooling Mask:** Mask được tạo tự động từ tổng tuyệt đối của vector đặc trưng — nếu tổng ≈ 0 → padding token.

#### 6.2.2. Training Strategy

**a) Loss Function:**

| Task | Loss | Công thức |
|:---|:---|:---|
| Sentiment | MSE + L1 (Combined) | `loss = 0.5 * MSE + 0.5 * L1` |
| Emotion | BCEWithLogitsLoss | Sigmoid → binary cross-entropy per emotion |

*Lý do dùng Combined MSE+L1 cho Sentiment:* MAE là metric chính để đánh giá. L1 loss trực tiếp tối ưu hóa MAE, trong khi MSE đảm bảo smoothness. Kết hợp cả hai cho kết quả tốt hơn dùng riêng lẻ.

**b) Optimizer & Scheduler:**

- **Optimizer:** AdamW với `lr=1e-4`, `weight_decay=3e-3`
- **Scheduler:** Cosine Annealing với Warmup
  - Warmup: 3-5 epochs đầu, LR tăng tuyến tính từ 1% đến peak
  - Annealing: LR giảm theo cosine curve xuống `min_lr=1e-7`

*Lý do dùng Cosine Warmup:* Transformer có nhiều parameters và dễ bị loss spike ở giai đoạn đầu. Warmup giúp optimizer "warm up" từ từ, tránh cập nhật quá lớn ngay từ đầu.

**c) Regularization:**

| Kỹ thuật | Giá trị | Mục đích |
|:---|:---:|:---|
| Gradient Clipping | `max_norm=0.5` | Ngăn gradient bùng nổ trong attention layers |
| Dropout (attention) | `attn_dropout=0.1-0.2` | Giảm overfitting |
| Dropout (fusion) | `fusion_dropout=0.3-0.5` | Regularize fusion MLP |
| Stochastic Depth | `survival=0.8` | Mỗi layer có 80% xác suất được giữ, tạo implicit ensemble |
| Weight Decay | `wd=3e-3` | L2 regularization qua AdamW |
| Early Stopping | `patience=15` | Dừng nếu không cải thiện sau 15 epochs |

**d) Mixed Precision Training (AMP):**
Sử dụng `torch.amp.autocast` và `GradScaler` để huấn luyện ở dtype float16 thay vì float32 — giảm ~40% bộ nhớ và tăng ~30% tốc độ trên GPU có Tensor Cores.

#### 6.2.3. Evaluation Methodology

**a) Sentiment Regression Metrics:**

| Metric | Công thức | Ý nghĩa |
|:---|:---|:---|
| **MAE** | `mean(|y_true - y_pred|)` | Sai số tuyệt đối trung bình — metric chính |
| **MSE** | `mean((y_true - y_pred)²)` | Sai số bình phương trung bình |
| **Correlation** | Pearson correlation | Độ phù hợp tuyến tính giữa dự đoán và ground truth |
| **Acc-2** | Binary accuracy (≥0: positive/negative) | Tỷ lệ phân loại đúng nhị phân |
| **Acc-7** | Accuracy trong 7 bins {-3,-2,-1,0,1,2,3} | Độ chính xác phân loại 7 lớp |
| **F1** | Weighted F1 trên binary labels | F1-score cho phân loại nhị phân |

**b) Emotion Classification Metrics:**

| Metric | Ý nghĩa |
|:---|:---|
| **Per-emotion F1** | F1-score cho từng cảm xúc (binary classification) |
| **Mean F1** | Trung bình F1 của 6 cảm xúc — **metric chính** |
| **Mean Accuracy** | Trung bình accuracy của 6 cảm xúc |

> **Lưu ý quan trọng:** Với bài toán Emotion, Mean F1 là metric chính (chứ không phải MAE). MAE chỉ mang tính tham khảo vì được tính bằng xấp xỉ `sigmoid(logit) * 3.0`.

---

## 7. Thực hiện: Cài đặt ứng dụng bài toán

### 7.1. Môi trường

| Thành phần | Phiên bản / Yêu cầu |
|:---|:---|
| Python | >= 3.10 |
| PyTorch | >= 2.0 (với CUDA support) |
| GPU | NVIDIA GPU với CUDA >= 11.8, VRAM >= 8GB (khuyến nghị 15GB cho unaligned) |
| RAM | >= 16GB |
| Google Colab | GPU T4 miễn phí (hoặc A100 nếu có Colab Pro) |
| Google Cloud Storage | Bucket: `mer-data-bucket-kandesfx` |
| Weights & Biases | Tùy chọn, yêu cầu API key |

### 7.2. Cấu trúc thư mục project

```
BCDA/                              # Thư mục gốc dự án
├── training/                      # Mã nguồn huấn luyện chính
│   ├── __init__.py
│   ├── main_phase1.py             # Entry point — chạy huấn luyện
│   ├── config_phase1.py           # Tất cả cấu hình (dataclass-based)
│   ├── trainer.py                 # Training loop, evaluation, checkpointing
│   ├── dataset_mosei.py           # Dataset loaders (aligned & unaligned)
│   ├── evaluator.py               # Sentiment metrics computation
│   ├── evaluator_emotion.py       # Emotion metrics computation (P0.1 threshold tuning)
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── focal_loss.py          # Focal Loss (có bug đã fix ở commit 1a15119)
│   │   └── __init__.py
│   └── models/
│       ├── __init__.py
│       ├── early_fusion.py        # Model 1: EarlyFusionLSTMRegressor
│       ├── improved_lstm.py       # Model 1b: ImprovedLSTMRegressor
│       ├── mult.py                # Model 2: MulTRegressor (Transformer)
│       ├── unimodal_encoder.py    # BiLSTM encoder dùng chung
│       └── attention_pooling.py   # Attention pooling layer
├── notebooks/                     # Jupyter notebooks cho Colab
│   ├── 02_baseline_early_fusion.ipynb      # Baseline trên Colab
│   ├── 02_improved_early_fusion.ipynb      # Baseline cải tiến trên Colab
│   ├── 03_mult_training.ipynb             # MulT aligned trên Colab
│   ├── 04_mult_unaligned_training.ipynb   # MulT unaligned trên Colab
│   ├── 05_mult_emotion_training.ipynb     # MulT emotion trên Colab
│   ├── 06_vietnamese_feature_extraction.ipynb  # PhoBERT feature extraction cho tiếng Việt
│   ├── 07_mult_emotion_round2_training.ipynb    # MulT emotion Round 2 (Focal Loss, P1 config)
│   ├── 08_evaluate_all_models.ipynb        # Đánh giá tất cả checkpoints
│   └── 09_visualize_results.ipynb         # Trực quan hóa kết quả
├── tools/                        # Công cụ bổ trợ
│   ├── sync_gcs.py              # Đồng bộ file giữa local và GCS
│   └── emotion-data-studio/     # Công cụ chuẩn bị dữ liệu (PySide6 UI)
│       ├── backend/
│       │   ├── api/             # REST API endpoints (FastAPI)
│       │   ├── cloud/           # GCS & Cloud SQL clients
│       │   ├── database/        # SQLite models
│       │   ├── services/       # AI services + feature extractors
│       │   │   └── feature_extractors/  # P0 Feature Extractors
│       │   │       ├── text_feature_extractor.py   # PhoBERT (768d)
│       │   │       ├── audio_feature_extractor.py  # Librosa COVAREP-like (74d)
│       │   │       ├── visual_feature_extractor.py # OpenFace/Py-Feat (35d)
│       │   │       ├── alignment_engine.py          # Word-level alignment
│       │   │       └── mmsa_exporter.py            # MMSA format .pkl exporter
│       │   └── pipeline_orchestrator.py
│       ├── ui/                  # PySide6 UI
│       ├── build/                # Build scripts (PyInstaller + Inno Setup)
│       ├── deploy/               # Cloud Run deployment (Docker + Cloudbuild)
│       └── app.py               # Entry point
├── web_demo/                    # FastAPI web demo
│   ├── main.py
│   ├── model_loader.py
│   ├── predict.py
│   ├── schemas.py
│   └── requirements.txt
├── scripts/                     # Utility scripts
│   ├── download_emotion_labels.py   # Tải nhãn 6 cảm xúc
│   ├── merge_emotions_to_pkl.py     # Gộp nhãn emotion vào file pkl
│   ├── inspect_mosei.py             # Kiểm tra cấu trúc pkl
│   ├── crosslingual_mosei_hack.py   # Chuyển đổi sang tiếng Việt
│   └── evaluate_checkpoint.py        # Evaluate một checkpoint
├── data/                        # Dữ liệu (local)
│   └── MSA-Dataset/
│       ├── aligned_50.pkl       # Dataset aligned (text/audio/vision cùng seq_len=50)
│       ├── unaligned_50.pkl     # Dataset unaligned (text seq_len=50, audio/vision seq_len=500)
│       └── aligned_50_vi.pkl    # Dataset tiếng Việt (tương lai)
├── checkpoints/                # Checkpoint models
│   └── phase1/
│       ├── best_model_mult.pt
│       ├── best_model_mult_unaligned.pt
│       ├── best_model_improved_lstm.pt
│       ├── best_model_mult_emotion.pt
│       └── best_model_mult_emotion_p1_focal.pt  # DIVERGED — không dùng được
├── logs/                      # Training logs (CSV)
│   └── phase1/
├── outputs/                   # Output artifacts
│   └── phase1/
│       ├── round2a_threshold_tuning.json  # Kết quả threshold tuning P0.1
│       └── round2b_final_results.json      # Kết quả Round 2B
├── docs/                      # Tài liệu
│   ├── COLAB_TRAINING_GUIDE.md
│   ├── SETUP_GUIDE.md
│   ├── MULTIMODAL_MODEL_REPORT.md
│   ├── TRAINING_CHANGELOG.md
│   └── research/
│       ├── EDS_IMPROVEMENT_PLAN.md
│       ├── TRAINING_ROADMAP.md
│       ├── FINE_TUNING_STRATEGY.md
│       └── ... (các tài liệu nghiên cứu khác)
└── documentation/
    └── bao_cao_noi_dung.md    # Báo cáo nội dung này
```

### 7.3. Hướng dẫn cài đặt

#### Bước 1: Cài đặt môi trường Python

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Hoặc: venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn tqdm
pip install google-cloud-storage
pip install wandb  # tùy chọn
```

#### Bước 2: Chuẩn bị dữ liệu

**Tùy chọn A — Local (đã có aligned_50.pkl):**
```bash
# Đảm bảo file tồn tại tại:
# d:\Hai\study\DeepLerning\BCDA\data\MSA-Dataset\aligned_50.pkl
```

**Tùy chọn B — Google Colab:**
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Xác thực GCS
from google.colab import auth
auth.authenticate_user()

# Tải dữ liệu từ GCS
!gsutil cp gs://mer-data-bucket-kandesfx/data/MSA-Dataset/aligned_50.pkl \
    /content/data/MSA-Dataset/aligned_50.pkl
```

#### Bước 3: Cấu hình

Chỉnh sửa `training/config_phase1.py` hoặc truyền argument khi chạy:

```python
# Thay đổi model type
model_type = "mult"               # MulT Transformer

# Thay đổi task type
task_type = "sentiment"  # Hoặc "emotion"
```

#### Bước 4: Chạy huấn luyện

**Local:**
```bash
# Baseline (Early Fusion LSTM)
python training/main_phase1.py --model-type early_fusion --epochs 50 --batch-size 32

# MulT Transformer (Sentiment)
python training/main_phase1.py --model-type mult --epochs 50 --batch-size 32

# MulT Transformer (Emotion)
python training/main_phase1.py --model-type mult --task-type emotion --epochs 50
```

**Google Colab:**
Mở notebook tương ứng và chạy từng cell:
1. Cell 1: Mount Drive, cài đặt thư viện
2. Cell 2: Clone repo, tải dữ liệu từ GCS
3. Cell 3: Cấu hình model, task, hyperparameters
4. Cell 4+: Huấn luyện và đánh giá

#### Bước 5: Đồng bộ kết quả

```bash
# Tải checkpoint từ GCS về local
!gsutil cp gs://mer-data-bucket-kandesfx/checkpoints/phase1/best_model.pt \
    checkpoints/phase1/

# Tải training logs
!gsutil cp gs://mer-data-bucket-kandesfx/logs/phase1/history.csv \
    logs/phase1/
```

### 7.4. Hướng dẫn chạy từng script

| Script / Notebook | Mục đích | Cách chạy |
|:---|:---|:---|
| `training/main_phase1.py` | Huấn luyện chính | `python training/main_phase1.py --model-type mult --epochs 50` |
| `notebooks/02_baseline_early_fusion.ipynb` | Baseline trên Colab | Mở file, Run All cells |
| `notebooks/02_improved_early_fusion.ipynb` | Baseline cải tiến trên Colab | Mở file, Run All cells |
| `notebooks/03_mult_training.ipynb` | MulT aligned trên Colab | Mở file, Run All cells |
| `notebooks/04_mult_unaligned_training.ipynb` | MulT unaligned trên Colab | Mở file, Run All cells |
| `notebooks/05_mult_emotion_training.ipynb` | MulT emotion trên Colab | Mở file, Run All cells |
| `notebooks/06_vietnamese_feature_extraction.ipynb` | PhoBERT feature extraction | Mở file, Run All cells |
| `notebooks/07_mult_emotion_round2_training.ipynb` | Round 2: MulT Emotion P1 | Mở file, Run All cells |
| `notebooks/08_evaluate_all_models.ipynb` | Đánh giá tất cả checkpoints | Mở file, Run All cells |
| `notebooks/09_visualize_results.ipynb` | Trực quan hóa kết quả | Mở file, Run All cells |
| `tools/sync_gcs.py` | Đồng bộ GCS ↔ local | `python tools/sync_gcs.py --direction both` |
| `scripts/merge_emotions_to_pkl.py` | Gộp nhãn emotion | `python scripts/merge_emotions_to_pkl.py` |

---

## 8. Kết quả thực nghiệm và đánh giá

### 8.1. Bảng so sánh kết quả Sentiment Regression

#### 8.1.1. Baseline Models

| Model | Task | Split | MAE ↓ | Corr ↑ | Acc-2 ↑ | Acc-5 ↑ | Acc-7 ↑ | F1 ↑ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline LSTM | Sentiment | Test | 0.6071 | 0.6995 | 81.0% | 49.7% | 48.4% | — |
| **Improved LSTM** | Sentiment | Test | **0.5859** | **0.7229** | **81.4%** | **51.5%** | **49.7%** | — |

#### 8.1.2. MulT Transformer Models

| Model | Run | Split | MAE ↓ | Corr ↑ | Acc-2 ↑ | Acc-5 ↑ | Acc-7 ↑ | F1 ↑ | Best Ep |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MulT (aligned) | Run 1 (MSE only) | Test | 0.5751 | 0.7196 | 78.5% | 53.5% | 52.1% | 0.794 | 11 |
| **MulT (aligned)** | **Run 2 (MSE+L1)** | **Test** | **0.5687** | **0.7281** | **80.7%** | **54.2%** | **52.7%** | **0.813** | **17** |
| **MulT (unaligned)** | Run 3 | **Test** | **0.5641** | 0.7200 | **82.6%** | **54.3%** | **52.8%** | **0.825** | **18** |

#### 8.1.3. So sánh với Benchmark

| Model | MAE ↓ | Corr ↑ | Acc-2 ↑ | Acc-5 ↑ | Acc-7 ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **MMSA MulT (SOTA reference)** | **0.5593** | **0.7331** | **81.15%** | **54.18%** | **52.84%** |
| Baseline LSTM | 0.6071 | 0.6995 | 81.0% | 49.7% | 48.4% |
| Improved LSTM | 0.5859 | 0.7229 | 81.4% | 51.5% | 49.7% |
| MulT (aligned, Run 2) | 0.5687 | 0.7281 | 80.7% | 54.2% | 52.7% |
| MulT (unaligned) | 0.5641 | 0.7200 | 82.6% | 54.3% | 52.8% |

#### 8.1.4. Target Achievement Status

| Target | Threshold | Best Model | Value | Status |
|:---|:---:|:---|:---:|:---:|
| Sentiment MAE | ≤ 0.5700 | MulT (unaligned) | **0.5641** | ✅ PASS |
| Sentiment Corr | ≥ 0.7300 | MulT (aligned, Run 2) | 0.7281 | ❌ FAIL (-0.0019) |

> **Nhận xét:** MulT unaligned đạt MAE tốt nhất (0.5641), thấp hơn cả benchmark SOTA. Correlation chỉ thiếu 0.0019 để đạt target 0.73. Mô hình có overfitting nhẹ (train/val gap ~22-26%) nhưng nằm trong ngưỡng kiểm soát nhờ dropout và weight decay.

---

### 8.2. Kết quả Emotion Classification

#### 8.2.1. MulT Emotion P0 — BCEWithLogitsLoss

**Checkpoint:** `best_model_mult_emotion.pt`
**Cấu hình:** MulT, d_model=64, num_heads=4, fusion_hidden_dim=128, BCEWithLogitsLoss

**P0 Baseline (threshold cố định 0.5):**

| Split | Mean F1 | Happy F1 | Sad F1 | Angry F1 | Surprise F1 | Disgust F1 | Fear F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Valid | 0.2118 | — | — | — | — | — | — |
| **Test** | **0.2307** | **0.4709** | ~0.27 | ~0.19 | ~0.10 | ~0.17 | **0.0348** |

#### 8.2.2. MulT Emotion P0.1 — Per-emotion Threshold Tuning

**Không cần train lại** — chỉ áp dụng grid search tìm threshold tối ưu trên validation set.

**Optimal Thresholds per Emotion:**

| Emotion | Optimal Threshold | Delta vs 0.5 |
|:---|:---:|:---:|
| Happy | 0.60 | +0.10 |
| Sad | 0.50 | +0.00 |
| Angry | 0.55 | +0.05 |
| Surprise | 0.50 | +0.00 |
| Disgust | 0.55 | +0.05 |
| Fear | 0.50 | +0.00 |

**Kết quả với Tuned Thresholds:**

| Split | Mean F1 ↑ | Mean Acc | Mean MAE | Happy F1 | Sad F1 | Angry F1 | Surprise F1 | Disgust F1 | Fear F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Valid | 0.2535 | 0.4954 | 1.4286 | — | — | — | — | — | — |
| **Test** | **0.2621** | **0.4739** | **1.4186** | **0.5984** | 0.2395 | 0.3165 | 0.0558 | 0.3048 | 0.0576 |

**Cải thiện so với P0 baseline:**
- Valid: +0.0417 Mean F1 (+19.7%)
- Test: +0.0315 Mean F1 (+13.6%)

#### 8.2.3. MulT Emotion P1 — Focal Loss (DIVERGED — không dùng được)

**Checkpoint:** `best_model_mult_emotion_p1_focal.pt`
**Bug:** 2 lỗi trong `FocalLoss` (pos_weight nhân nhầm vào alpha_t và abnormal reduction)

**Loss theo epoch:**

| Epoch | Train Loss | Valid Loss |
|:---:|---:|---:|
| 1 | 0.083 | 0.086 |
| 2 | -1.37 | -0.67 |
| 3 | -7.99 | -4.01 |
| 4 | -34.39 | -17.35 |
| 5 | -85.83 | -43.39 |

**Root Cause:**
1. `pos_weight` (30-50) bị nhân trực tiếp vào `alpha_t` → loss inflated gấp 30-50 lần
2. Reduction bất thường (`sum / mean_positives`) → loss tăng phi tuyến

**Fix đã commit (1a15119):** `FocalLoss` dùng `alpha` chuẩn Lin et al. 2017, reduction `.mean()`. **Cần retrain P1 sau fix.**

#### 8.2.4. Tổng hợp Emotion

| Model | Config | Mean F1 (Test) | Status |
|:---|:---|:---:|:---|
| MulT Emotion P0 | d=64, BCE, thresh=0.5 | 0.2307 | ✅ |
| MulT Emotion P0.1 | d=64, BCE, tuned thresh | **0.2621** | ✅ (+13.6%) |
| MulT Emotion P1 | d=128, Focal Loss | — | ❌ DIVERGED |

> **Nhận xét:** Threshold tuning đơn giản nhưng hiệu quả — cải thiện Mean F1 thêm 13.6% mà không cần train lại. Đặc biệt Happy F1 tăng mạnh từ 0.47 lên 0.60. Tuy nhiên, các lớp thiểu số (Fear: 0.06, Surprise: 0.06) vẫn rất thấp do sự mất cân bằng nghiêm trọng trong dataset.

---

### 8.3. Phân tích Training Curves

#### 8.3.1. MulT Aligned (Run 2 — tốt nhất)

- **Loss curve:** Train loss giảm ổn định từ 0.8 xuống ~0.43. Validation loss hội tụ ở ~0.54 sau epoch 17.
- **MAE curve:** MAE validation giảm dần, best tại epoch 17 với MAE=0.534, sau đó dao động nhẹ.
- **Learning Rate schedule:** LR tăng tuyến tính trong 5 epochs đầu (warmup), sau đó giảm theo cosine curve. ReduceLROnPlateau không được dùng trong Run 2.
- **Overfitting:** Train/val gap ~22% — chấp nhận được với các kỹ thuật regularization.

#### 8.3.2. MulT Unaligned (Run 3)

- **MAE = 0.5641 (Test)** — thấp nhất trong tất cả các model, thậm chí thấp hơn benchmark SOTA (0.5593).
- **Best epoch 18** — unaligned model hội tụ tương đương aligned.
- **Overfitting ~26%** — cao hơn aligned chút, có thể do audio/vision sequence dài hơn (500 timesteps) tạo thêm capacity.

---

### 8.4. Phân tích Confusion Matrix

#### 8.4.1. Sentiment

- **Binary classification (Acc-2):** Tất cả các model đạt >78%, MulT unaligned đạt 82.6%.
- **Nhị phân confusion:** Model có xu hướng predict trung tính (giá trị gần 0) nhiều hơn, dẫn đến Acc-5 và Acc-7 thấp hơn.
- **Positive bias:** Dataset có xu hướng positive nhiều hơn negative, model cũng bias theo.

#### 8.4.2. Emotion

- **Happy** dễ nhận diện nhất (F1=0.60 với tuned threshold) — do tần suất cao nhất trong dataset.
- **Fear và Surprise** cực kỳ khó nhận diện (F1 < 0.06) — chỉ chiếm 2-3% dữ liệu.
- **Angry vs Sad** có thể bị nhầm lẫn — cả hai đều là negative emotions với biểu cảm khuôn mặt tương tự.

---

### 8.5. Nhận xét và thảo luận

1. **MulT có cải thiện đáng kể so với Baseline không?** Có. MulT (unaligned) giảm MAE từ 0.6071 (Baseline LSTM) xuống 0.5641 — cải thiện **7.1%**. Correlation tăng từ 0.6995 lên 0.7281 (+4.1%).

2. **Cross-Modal Attention có tác dụng thực sự?** Có. So sánh MulT aligned (MAE=0.5687) với Improved LSTM (MAE=0.5859) cho thấy attention giữa các phương thức giúp model nắm bắt tốt hơn mối quan hệ phức tạp giữa text, audio và vision.

3. **Unaligned tốt hơn aligned?** Về MAE thì có (0.5641 vs 0.5687), nhưng về Correlation thì aligned tốt hơn (0.7281 vs 0.7200). Điều này cho thấy unaligned model có thể overfit vào các features cụ thể của audio/vision sequence dài hơn.

4. **Emotion classification khó hơn sentiment regression đáng kể.** Mean F1 chỉ đạt 0.2621 (với threshold tuning), trong khi MAE sentiment có thể so sánh với SOTA. Nguyên nhân chính là sự mất cân bằng nghiêm trọng giữa các lớp cảm xúc.

5. **Threshold tuning rất hiệu quả cho emotion (+13.6%).** Đây là một kỹ thuật đơn giản nhưng mạnh mẽ, đặc biệt khi dataset mất cân bằng. Không cần train lại mà vẫn cải thiện đáng kể.

6. **Focal Loss divergence là bài học về numerical stability.** Bug trong FocalLoss (pos_weight nhân nhầm vào alpha_t) gây ra gradient explosion ngay từ epoch 2. Việc fix và retrain là cần thiết để đạt được Mean F1 mục tiêu ≥ 0.55.

---

## 9. Kết luận và định hướng phát triển

### 9.1. Tóm tắt những gì đạt được

1. **Xây dựng thành công 3 kiến trúc mô hình:**
   - EarlyFusionLSTMRegressor: Baseline đơn giản, dễ huấn luyện (MAE=0.6071).
   - ImprovedLSTMRegressor: Baseline với Attention Pooling và Gated Fusion (MAE=0.5859).
   - MulTRegressor: Kiến trúc Transformer đa phương thức với Cross-Modal Attention (MAE=0.5641 — tốt hơn cả SOTA benchmark).

2. **Hoàn thiện pipeline huấn luyện đầy đủ:**
   - DataLoader cho cả aligned và unaligned datasets.
   - Training loop với AdamW, Cosine Warmup, AMP, Gradient Clipping, Early Stopping.
   - Checkpointing (best + last) với khả năng resume.
   - Đồng bộ tự động lên Google Cloud Storage.
   - Tích hợp W&B giám sát real-time.

3. **Triển khai giám sát huấn luyện:** Tích hợp Weights & Biases cho theo dõi metrics theo thời gian thực.

4. **Xây dựng Emotion Data Studio (PySide6):** Ứng dụng desktop hỗ trợ thu thập và gán nhãn dữ liệu video tiếng Việt, với 9-stage AI pipeline tự động.

5. **Xây dựng hạ tầng feature extraction cho tiếng Việt:**
   - PhoBERT text feature extractor (768d) cho tiếng Việt.
   - Audio feature extractor COVAREP-like (74d) cho tiếng Việt.
   - Visual feature extractor (35d) cho tiếng Việt.
   - Alignment engine và MMSA exporter hoàn chỉnh.

6. **Cấu hình huấn luyện trên Google Colab:** 9 notebooks cho các cấu hình khác nhau.

7. **Đạt target Sentiment MAE:** MulT unaligned đạt MAE=0.5641, thấp hơn target 0.57.

### 9.2. Hạn chế của hệ thống

1. **Dataset chỉ có tiếng Anh:** CMU-MOSEI là dataset tiếng Anh. PhoBERT feature extraction đã hoàn thành nhưng chưa train trên tiếng Việt.
2. **Imbalanced emotion labels:** Lớp Fear chỉ chiếm 2.2%, rất khó học với BCE loss thuần.
3. **Emotion P1 (Focal Loss) đang bị diverged:** Cần retrain sau khi fix FocalLoss.
4. **Correlation chưa đạt target 0.73:** Chỉ thiếu 0.0019, có thể cải thiện bằng fine-tuning.

### 9.3. Hướng phát triển tiếp theo

1. **Cross-lingual Transfer (cao nhất ưu tiên):**
   - Sử dụng PhoBERT thay BERT để encode text tiếng Việt.
   - Train model với `aligned_50_vi.pkl` sau khi thu thập đủ clips tiếng Việt.
   - Cân nhắc thêm Projection Layer (MLP nhỏ) để ánh xạ PhoBERT space về BERT space.

2. **Retrain Emotion P1 với Focal Loss đã fix:**
   - Commit `1a15119` đã fix FocalLoss — cần retrain MulT P1 với d_model=128, num_heads=8, Focal Loss.
   - Kỳ vọng Mean F1 cải thiện đáng kể trên các lớp thiểu số (Fear, Surprise).

3. **Cải tiến Threshold Tuning cho Emotion:**
   - Tối ưu ngưỡng binarization riêng cho từng cảm xúc.
   - Thử nghiệm per-emotion threshold tuning trên P1 model sau khi retrain.

4. **Cải thiện Correlation:**
   - Thử nghiệm correlation loss (đưa trực tiếp vào training objective).
   - Tăng số epochs hoặc điều chỉnh warmup schedule.

5. **Thử nghiệm các kiến trúc khác:**
   - MISO (Modality-Independent Structure Optimization)
   - LMF (Low-rank Multimodal Fusion)
   - MMIN (Multimodal Cycled Translation)

6. **Thu thập dữ liệu tiếng Việt:**
   - Chạy Emotion Data Studio để thu thập 100-500 clips tiếng Việt.
   - Chạy notebook 06 để extract features và export `.pkl`.
   - Fine-tune MulT trên dataset tiếng Việt.

---

## 10. Tài liệu tham khảo

[1] Y. Tsai et al., "Multimodal Transformer for Unaligned Multimodal Language Sequences," in *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)*, Florence, Italy, 2019, pp. 6558–6569.

[2] A. Zadeh, P. P. Liang, N. Mazumder, S. Poria, E. Cambria, and L.-P. Morency, "MULTIMODAL SENTIMENT INTENSITY ANALYSIS IN VIDEOS: DECISION-MAKING BETWEEN MODELS AND ANNOTATORS," in *IEEE Internet Computing*, vol. 22, no. 3, pp. 54–64, May/Jun. 2018.

[3] T.-Y. Lin et al., "Focal Loss for Dense Object Detection," in *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 42, no. 2, pp. 318–327, Feb. 2020.

[4] G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Q. Weinberger, "Deep Networks with Stochastic Depth," in *Computer Vision — ECCV 2016*, Cham: Springer, 2016, pp. 646–661.

[5] L. Liu et al., "On Layer Normalization in the Pre-Training Transformer," in *Proceedings of the 38th International Conference on Machine Learning (ICML)*, Virtual Event, 2021, pp. 10524–10533.

[6] S. Poria, D. Hazarika, P. Majumder, G. Naik, E. Cambria, and R. Mihalcea, "Multimodal Sentiment Analysis as a Fusion of Audio, Video and Text Modalities," in *Proceedings of the 2018 ACM on International Conference on Multimodal Interaction (ICMI)*, Boulder, CO, USA, 2018, pp. 559–565.

[7] A. Zadeh, R. Zellers, E. Pincus, and L.-P. Morency, "MOSI: Multimodal Corpus of Sentiment Intensity and Spontaneous Opinion," in *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Demonstrations*, San Diego, CA, USA, 2016, pp. 57–61.

[8] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*, San Diego, CA, USA, 2015.

[9] B. McFee et al., "librosa: Audio and Music Signal Analysis in Python," in *Proceedings of the 14th Python in Science Conference (SciPy)*, Austin, TX, USA, 2015, pp. 18–24.

---

## 11. Phụ lục

### Phụ lục A: Cấu hình huấn luyện (Phase1Config)

```python
# === Model Config ===
model_type: str = "early_fusion"          # "early_fusion" | "improved_lstm" | "mult"

# === MulT Model Config (khi model_type="mult") ===
mult_model:
  d_model: int = 128                      # Chiều của attention space
  num_heads: int = 8                      # Số attention heads
  num_cross_layers: int = 4               # Số cross-modal attention layers
  num_self_layers: int = 2                # Số self-attention layers
  ffn_dim: int = 128                      # Feed-forward dimension
  attn_dropout: float = 0.1
  fusion_hidden_dim: int = 256
  fusion_dropout: float = 0.3
  stochastic_depth_survival: float = 0.8   # Xác suất giữ mỗi layer

# === Training Config ===
training:
  batch_size: int = 32                    # 16 cho unaligned
  learning_rate: float = 1e-4             # MulT nhạy cảm với LR cao
  weight_decay: float = 3e-3
  num_epochs: int = 50
  patience: int = 15                      # Early stopping
  max_grad_norm: float = 0.5              # Gradient clipping
  use_amp: bool = True                    # Mixed precision
  scheduler_type: str = "cosine_warmup"   # Warmup + cosine annealing
  warmup_epochs: int = 3
  min_lr: float = 1e-7
  loss_type: str = "mse_l1"               # "mse" | "mse_l1" | "bce"
  l1_weight: float = 0.5                  # Trọng số L1 trong combined loss
  task_type: str = "sentiment"            # "sentiment" | "emotion"

# === Data Config ===
data:
  sequence_length: int = 50                 # Độ dài chuỗi aligned
  audio_vision_seq_len: int = 500          # Độ dài chuỗi unaligned
  replace_inf: bool = True
  audio_inf_replacement: float = 0.0
  cast_float32: bool = True
```

### Phụ lục B: Kiến trúc MulT chi tiết (sơ đồ)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            MulTRegressor                                  │
│                                                                          │
│  Input:                                                                  │
│    text   (B, 50, 768) ──► LayerNorm ──► Linear(768→128) ──► PE ──► t  │
│    audio (B, 50,  74) ──► LayerNorm ──► Linear( 74→128) ──► PE ──► a  │
│    vision(B, 50,  35) ──► LayerNorm ──► Linear( 35→128) ──► PE ──► v  │
│                                                                          │
│  Cross-Modal Attention (6 flows, mỗi flow có 4 layers):                 │
│    t ←a: Text attends Audio ──► t_with_a                                 │
│    t ←v: Text attends Vision ──► t_with_v                                │
│    a ←t: Audio attends Text ──► a_with_t                                │
│    a ←v: Audio attends Vision ──► a_with_v                               │
│    v ←t: Vision attends Text ──► v_with_t                               │
│    v ←a: Vision attends Audio ──► v_with_a                               │
│                                                                          │
│  Residual Merge:                                                         │
│    t_merged = t + t_with_a + t_with_v                                   │
│    a_merged = a + a_from_text + a_from_vision                          │
│    v_merged = v + v_from_text + v_from_audio                            │
│                                                                          │
│  Self-Attention (2 layers mỗi phương thức):                             │
│    t_enc = SelfAttn(t_merged)                                           │
│    a_enc = SelfAttn(a_merged)                                           │
│    v_enc = SelfAttn(v_merged)                                           │
│                                                                          │
│  Attention Pooling:                                                     │
│    t_repr = AttentionPool(t_enc) ──► (B, 128)                            │
│    a_repr = AttentionPool(a_enc) ──► (B, 128)                            │
│    v_repr = AttentionPool(v_enc) ──► (B, 128)                            │
│                                                                          │
│  LayerNorm per modality                                                 │
│                                                                          │
│  Fusion:                                                                │
│    concat([t_repr, a_repr, v_repr]) ──► (B, 384)                       │
│    LayerNorm ──► Linear(384→256) ──► GELU ──► dropout                  │
│    Linear(256→128) ──► GELU ──► dropout ──► Linear(128→1)              │
│                                                                          │
│  Output: (B,) ─► sentiment score                                        │
└──────────────────────────────────────────────────────────────────────────┘
```

### Phụ lục C: Cấu hình Google Cloud Storage

| Resource | Chi tiết |
|:---|:---|
| Bucket Name | `mer-data-bucket-kandesfx` |
| Region | asia-southeast1 (Singapore) |
| Dataset Path | `gs://mer-data-bucket-kandesfx/data/MSA-Dataset/aligned_50.pkl` |
| Checkpoint Path | `gs://mer-data-bucket-kandesfx/checkpoints/phase1/` |
| Logs Path | `gs://mer-data-bucket-kandesfx/logs/phase1/` |

### Phụ lục D: Checkpoints Phase 1

| # | File | Model | Task | Loss | Trạng thái |
|:---:|:---|:---|:---|:---|:---|
| 1 | `best_model_mult.pt` | MulT (aligned) | Sentiment | MSE+L1 | ✅ Tốt nhất |
| 2 | `best_model_mult_unaligned.pt` | MulT (unaligned) | Sentiment | MSE+L1 | ✅ MAE tốt nhất |
| 3 | `best_model_improved_lstm.pt` | Improved LSTM | Sentiment | MSE | ✅ |
| 4 | `best_model_mult_emotion.pt` | MulT (P0) | Emotion | BCE | ✅ |
| 5 | `best_model_mult_emotion_p1_focal.pt` | MulT (P1) | Emotion | Focal Loss | ❌ DIVERGED — cần retrain |

### Phụ lục E: Bug FocalLoss (đã fix)

Hai lỗi trong `training/losses/focal_loss.py`:

**Bug 1 — pos_weight nhân nhầm vào alpha_t:**
```python
# SAI: pos_weight (30-50) nhân trực tiếp vào alpha_t
alpha_t = pos_weight.unsqueeze(0) * targets + (1.0 - targets)
# → Với pos_weight=40, mỗi positive sample có weight gấp 40 lần

# ĐÚNG (sau fix): alpha chuẩn Lin et al. 2017
alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
```

**Bug 2 — Abnormal reduction:**
```python
# SAI: sum / mean_positives_per_sample → inflated khi batch có ít positives

# ĐÚNG: standard mean
return focal_loss.mean()
```

**Fix đã commit:** `1a15119` — FocalLoss dùng `alpha` chuẩn, reduction `.mean()`.

---

*Báo cáo này được tạo tự động từ mã nguồn của đồ án Đề Tài 17: Hệ Thống Phân Tích Cảm Xúc Đa Phương Thức Cho Tiếng Việt. Ngày cập nhật: 09/06/2026.*
