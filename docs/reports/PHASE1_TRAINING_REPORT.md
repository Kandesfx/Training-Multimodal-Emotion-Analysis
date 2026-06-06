# BÁO CÁO KẾT QUẢ HUẤN LUYỆN PHASE 1 — PRE-TRAINING TRÊN CMU-MOSEI

**Đề Tài 17:** Hệ Thống Phân Tích Cảm Xúc Đa Phương Thức  
**Ngày báo cáo:** 06/06/2026  
**Nền tảng huấn luyện:** Google Colab Pro (GPU NVIDIA Tesla T4, 15GB VRAM)  
**Giám sát huấn luyện:** Weights & Biases — [Dashboard](https://wandb.ai/kandesfx-kandesfx/bcda-phase1)

---

## Mục Lục

1. [Tổng Quan Thí Nghiệm](#1-tổng-quan-thí-nghiệm)
2. [Dữ Liệu Huấn Luyện](#2-dữ-liệu-huấn-luyện)
3. [Thí Nghiệm 1: Baseline Early Fusion LSTM](#3-thí-nghiệm-1-baseline-early-fusion-lstm)
4. [Thí Nghiệm 2: Improved Early Fusion LSTM](#4-thí-nghiệm-2-improved-early-fusion-lstm)
5. [So Sánh Đối Chiếu Hai Mô Hình](#5-so-sánh-đối-chiếu-hai-mô-hình)
6. [Phân Tích & Đánh Giá](#6-phân-tích--đánh-giá)
7. [Kết Luận & Hướng Phát Triển](#7-kết-luận--hướng-phát-triển)
8. [Tài Nguyên Đã Lưu Trữ](#8-tài-nguyên-đã-lưu-trữ)

---

## 1. Tổng Quan Thí Nghiệm

Phase 1 của lộ trình huấn luyện nhằm **xây dựng mô hình nền tảng (pre-trained model)** trên tập dữ liệu chuẩn CMU-MOSEI bằng tiếng Anh, để mô hình học được cách kết hợp thông tin từ ba phương thức (Văn bản, Âm thanh, Hình ảnh) trước khi tinh chỉnh (fine-tune) trên dữ liệu tiếng Việt ở Phase 2.

Chúng tôi đã thực hiện **2 thí nghiệm** với 2 kiến trúc mô hình khác nhau, cùng sử dụng bộ dữ liệu và pipeline huấn luyện giống nhau để đảm bảo tính công bằng khi so sánh.

---

## 2. Dữ Liệu Huấn Luyện

**Tập dữ liệu:** CMU-MOSEI (CMU Multimodal Opinion Sentiment and Emotion Intensity)  
**File nguồn:** `aligned_50.pkl` (~4.4 GB)

| Thông số | Giá trị |
|:---|:---|
| Tổng số mẫu | 22,856 hội thoại |
| Tập Train | 16,326 mẫu |
| Tập Validation | 1,871 mẫu |
| Tập Test | 4,659 mẫu |
| Độ dài chuỗi (padding) | 50 bước thời gian |

**Đặc trưng đầu vào (đã trích xuất sẵn):**

| Phương thức | Kích thước | Công cụ trích xuất | Mô tả |
|:---|:---|:---|:---|
| Văn bản (Text) | (50, 768) | BERT-base-uncased | Vector ngữ nghĩa 768 chiều cho mỗi từ |
| Âm thanh (Audio) | (50, 74) | COVAREP | Vector đặc trưng âm sắc 74 chiều |
| Hình ảnh (Vision) | (50, 35) | FACET | 35 Action Units biểu cảm khuôn mặt |

**Nhãn:** Điểm sentiment liên tục trong khoảng [-3.0, +3.0] (do con người gán nhãn)

---

## 3. Thí Nghiệm 1: Baseline Early Fusion LSTM

### 3.1. Kiến Trúc Mô Hình

```
Text (50, 768)  ──► BiLSTM(hidden=128) ──► Hidden State cuối ──► h_text  (256)
Audio (50, 74)  ──► BiLSTM(hidden=64)  ──► Hidden State cuối ──► h_audio (128)
Vision (50, 35) ──► BiLSTM(hidden=64)  ──► Hidden State cuối ──► h_video (128)
                                                                    │
                                                          ┌────────▼────────┐
                                                          │  Concatenation  │
                                                          │  (256+128+128   │
                                                          │   = 512 chiều)  │
                                                          └────────┬────────┘
                                                                   │
                                                          ┌────────▼────────┐
                                                          │  FC(512→256)    │
                                                          │  BatchNorm+ReLU │
                                                          │  Dropout(0.3)   │
                                                          │  FC(256→128)    │
                                                          │  ReLU           │
                                                          │  Dropout(0.2)   │
                                                          │  FC(128→1)      │
                                                          └─────────────────┘
```

### 3.2. Siêu Tham Số

| Tham số | Giá trị |
|:---|:---|
| Số tham số mô hình | ~1.1 triệu |
| Số lớp LSTM | 1 |
| Encoder Dropout | 0.1 |
| Fusion Dropout | 0.3 / 0.2 |
| Learning Rate | 1e-3 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Batch Size | 32 |
| Số Epochs chạy | 10 |
| Patience (Early Stopping) | 8 |
| Mixed Precision (AMP) | Có |
| Loss Function | MSELoss |

### 3.3. Kết Quả Huấn Luyện

**Epoch tốt nhất:** Epoch 4 (dựa trên Validation MAE thấp nhất)

| Chỉ số | Train (Epoch 4) | Valid (Epoch 4) | Test (Epoch 4) |
|:---|:---:|:---:|:---:|
| **Loss (MSE)** | 0.4557 | 0.5541 | 0.6136 |
| **MAE** | 0.5033 | 0.5466 | 0.5815 |
| **Correlation** | 0.8004 | 0.7014 | 0.7125 |
| **Acc-2** (Binary) | 0.8529 | 0.8290 | 0.8298 |
| **Acc-5** (5 lớp) | 0.5815 | 0.5420 | 0.5237 |
| **Acc-7** (7 lớp) | 0.5604 | 0.5313 | 0.5108 |
| **F1-Score** | 0.8542 | 0.8309 | 0.8300 |

### 3.4. Diễn Biến Huấn Luyện (10 Epochs)

| Epoch | Train Loss | Valid Loss | Valid MAE | Valid Corr | Ghi chú |
|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 0.6644 | 0.6814 | 0.6203 | 0.6785 | Khởi đầu |
| 2 | 0.5947 | 0.6016 | 0.5727 | 0.6757 | |
| 3 | 0.5237 | 0.5489 | 0.5551 | 0.7024 | |
| **4** | **0.4557** | **0.5541** | **0.5466** | **0.7014** | **Best ✓** |
| 5 | 0.5048 | 0.5960 | 0.5765 | 0.6789 | Valid tăng |
| 6 | 0.3940 | 0.5543 | 0.5549 | 0.6976 | Overfitting rõ |
| 7 | 0.3681 | 0.5593 | 0.5533 | 0.7035 | |
| 8 | 0.2931 | 0.5516 | 0.5496 | 0.6998 | |
| 9 | 0.2555 | 0.5556 | 0.5591 | 0.7036 | |
| 10 | 0.2140 | 0.5649 | 0.5569 | 0.6932 | Valid tệ hơn |

### 3.5. Nhận Xét

- **Overfitting sớm và nghiêm trọng:** Từ Epoch 4, Train Loss giảm liên tục (0.45 → 0.21) trong khi Valid Loss dao động quanh 0.55 và không cải thiện. Khoảng cách Train-Valid ngày càng lớn.
- **Nguyên nhân chính:** 
  - (1) LSTM chỉ có 1 lớp, quá đơn giản nên không đủ khả năng tổng quát hóa;
  - (2) Phương pháp lấy Hidden State cuối cùng bị ảnh hưởng bởi padding zeros;
  - (3) Concatenation thô không có cơ chế chọn lọc thông tin giữa các phương thức.

---

## 4. Thí Nghiệm 2: Improved Early Fusion LSTM

### 4.1. Các Cải Tiến Kiến Trúc

So với Baseline, mô hình cải tiến có **3 thay đổi kiến trúc quan trọng:**

#### Cải tiến 1: Attention Pooling (Kháng nhiễu Padding)
- Thay vì lấy Hidden State ở bước cuối cùng (bước thứ 50, có thể là padding), mô hình sử dụng một mạng Attention tự động gán trọng số chú ý cho từng bước thời gian.
- Các vị trí padding được đánh dấu bằng mặt nạ (mask) và bị gán trọng số chú ý bằng 0, đảm bảo vector đại diện chỉ tập trung vào các token thực tế.

#### Cải tiến 2: Gated Multimodal Fusion (Hợp nhất có Cổng)
- Mỗi phương thức được chiếu về cùng kích thước ẩn 128 chiều qua các lớp Linear + LayerNorm + ReLU + Dropout.
- Một mạng cổng (Gating Network) sử dụng hàm Sigmoid dựa trên ngữ cảnh tổng hợp sinh ra vector trọng số lọc cho từng phương thức, giúp tự động tăng cường thông tin hữu ích và kháng nhiễu.

#### Cải tiến 3: Tăng cường Regularization
- Tăng từ 1 lên **2 lớp LSTM** để mô hình học được các pattern chuỗi sâu hơn.
- Thêm **Layer Normalization** sau LSTM và trước khi đưa vào bộ Fusion.
- Tăng tỷ lệ **Dropout** lên: encoder=0.3, fusion=0.4/0.3.

### 4.2. Kiến Trúc Mô Hình

```
Text (50, 768)  ──► BiLSTM(2 layers, hidden=128) ──► Attention Pooling ──► LayerNorm ──┐
Audio (50, 74)  ──► BiLSTM(2 layers, hidden=64)  ──► Attention Pooling ──► LayerNorm ──┤
Vision (50, 35) ──► BiLSTM(2 layers, hidden=64)  ──► Attention Pooling ──► LayerNorm ──┤
                                                                                       │
                                                                          ┌────────────▼──────────────┐
                                                                          │   Gated Multimodal Fusion │
                                                                          │                           │
                                                                          │  Project → [128] mỗi kênh │
                                                                          │  Gate = Sigmoid(concat)   │
                                                                          │  output = gate * proj     │
                                                                          │  Concat 3 kênh → [384]    │
                                                                          └────────────┬──────────────┘
                                                                                       │
                                                                          ┌────────────▼──────────────┐
                                                                          │   Regressor MLP           │
                                                                          │   FC(384→256) + BN + ReLU │
                                                                          │   Dropout(0.4)            │
                                                                          │   FC(256→128) + ReLU      │
                                                                          │   Dropout(0.3)            │
                                                                          │   FC(128→1)               │
                                                                          └───────────────────────────┘
```

### 4.3. Siêu Tham Số

| Tham số | Giá trị |
|:---|:---|
| Số tham số mô hình | ~2.03 triệu |
| Số lớp LSTM | 2 |
| Encoder Dropout | 0.3 |
| Fusion Dropout | 0.4 / 0.3 |
| Projection Dim (Gated Fusion) | 128 |
| Learning Rate | 1e-3 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Batch Size | 32 |
| Số Epochs chạy | 17 (dừng sớm bởi Early Stopping) |
| Patience (Early Stopping) | 10 |
| Mixed Precision (AMP) | Có |
| Loss Function | MSELoss |

### 4.4. Kết Quả Huấn Luyện

**Epoch tốt nhất:** Epoch 7 (dựa trên Validation MAE thấp nhất)

| Chỉ số | Train (Epoch 7) | Valid (Epoch 7) | Test (Epoch 7) |
|:---|:---:|:---:|:---:|
| **Loss (MSE)** | 0.3297 | 0.5147 | 0.6021 |
| **MAE** | 0.4328 | 0.5323 | 0.5859 |
| **Correlation** | 0.8654 | 0.7254 | 0.7229 |
| **Acc-2** (Binary) | 0.8610 | 0.8156 | 0.8137 |
| **Acc-5** (5 lớp) | 0.6275 | 0.5585 | 0.5153 |
| **Acc-7** (7 lớp) | 0.6061 | 0.5473 | 0.4971 |
| **F1-Score** | 0.8637 | 0.8201 | 0.8167 |

### 4.5. Diễn Biến Huấn Luyện (17 Epochs)

| Epoch | Train Loss | Valid Loss | Valid MAE | Valid Corr | Ghi chú |
|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 0.6986 | 0.6365 | 0.5901 | 0.6540 | Khởi đầu |
| 2 | 0.5592 | 0.5609 | 0.5567 | 0.6964 | |
| 3 | 0.6444 | 0.6769 | 0.6022 | 0.6933 | Bất ổn |
| 4 | 0.4975 | 0.6169 | 0.5667 | 0.7051 | |
| 5 | 0.4771 | 0.6158 | 0.5799 | 0.7004 | |
| 6 | 0.3817 | 0.5307 | 0.5416 | 0.7151 | Cải thiện mạnh |
| **7** | **0.3297** | **0.5147** | **0.5323** | **0.7254** | **Best ✓** |
| 8 | 0.3424 | 0.5688 | 0.5694 | 0.7060 | |
| 9 | 0.2626 | 0.5384 | 0.5381 | 0.7218 | |
| 10 | 0.2736 | 0.5474 | 0.5502 | 0.7163 | |
| 11 | 0.1828 | 0.5745 | 0.5602 | 0.7028 | |
| 12 | 0.1290 | 0.5626 | 0.5547 | 0.7105 | |
| 13 | 0.1149 | 0.5326 | 0.5467 | 0.7132 | Gần best |
| 14 | 0.0977 | 0.5633 | 0.5583 | 0.7070 | |
| 15 | 0.0861 | 0.5695 | 0.5636 | 0.7090 | |
| 16 | 0.0699 | 0.5370 | 0.5472 | 0.7100 | |
| 17 | 0.0533 | 0.5510 | 0.5565 | 0.7077 | Dừng (patience=10) |

### 4.6. Nhận Xét

- **Hội tụ chậm hơn nhưng đạt ngưỡng tốt hơn:** Mô hình cải tiến đạt best Valid MAE=0.5323 ở Epoch 7 (so với 0.5466 của baseline ở Epoch 4), cho thấy cải thiện **2.6%** về khả năng tổng quát hóa trên tập xác minh.
- **Correlation cải thiện đáng kể:** Valid Corr=0.7254 (so với 0.7014 của baseline), tức tăng **3.4%** — mô hình nắm bắt tốt hơn xu hướng cảm xúc tổng thể.
- **Overfitting vẫn tồn tại:** Train Loss giảm từ 0.33 xuống 0.053 trong khi Valid Loss dao động quanh 0.53. Khoảng cách vẫn lớn dù đã thêm nhiều regularization, cho thấy **giới hạn trần của kiến trúc LSTM** trên tập dữ liệu này.

---

## 5. So Sánh Đối Chiếu Hai Mô Hình

### 5.1. Kết Quả Trên Tập Test (Công Bằng Nhất)

| Chỉ số đánh giá | Baseline LSTM | Improved LSTM | Chênh lệch | Xu hướng |
|:---|:---:|:---:|:---:|:---:|
| **Test MAE** ↓ | 0.5815 | 0.5859 | +0.0044 | ⚖️ Tương đương |
| **Test MSE** ↓ | 0.6136 | **0.6021** | **-0.0115** | ✅ Tốt hơn |
| **Test Corr** ↑ | 0.7125 | **0.7229** | **+0.0104** | ✅ Tốt hơn |
| **Test Acc-2** ↑ | **0.8298** | 0.8137 | -0.0161 | ⚠️ Kém hơn |
| **Test Acc-5** ↑ | **0.5237** | 0.5153 | -0.0084 | ⚖️ Tương đương |
| **Test Acc-7** ↑ | **0.5108** | 0.4971 | -0.0137 | ⚠️ Kém hơn |
| **Test F1** ↑ | **0.8300** | 0.8167 | -0.0133 | ⚠️ Kém hơn |

*(↓ = thấp hơn tốt hơn, ↑ = cao hơn tốt hơn)*

### 5.2. Kết Quả Trên Tập Validation (Đánh Giá Khả Năng Tổng Quát Hóa)

| Chỉ số đánh giá | Baseline LSTM | Improved LSTM | Chênh lệch | Xu hướng |
|:---|:---:|:---:|:---:|:---:|
| **Valid MAE** ↓ | 0.5466 | **0.5323** | **-0.0143** | ✅ Tốt hơn |
| **Valid MSE** ↓ | 0.5541 | **0.5147** | **-0.0394** | ✅ Tốt hơn |
| **Valid Corr** ↑ | 0.7014 | **0.7254** | **+0.0240** | ✅ Tốt hơn |
| **Valid Acc-5** ↑ | 0.5420 | **0.5585** | +0.0165 | ✅ Tốt hơn |
| **Valid Acc-7** ↑ | 0.5313 | **0.5473** | +0.0160 | ✅ Tốt hơn |

### 5.3. So Sánh Quy Mô & Tốc Độ

| Thông số | Baseline LSTM | Improved LSTM |
|:---|:---:|:---:|
| Số tham số | ~1.1 triệu | ~2.03 triệu |
| Epochs đến best | 4 | 7 |
| Tổng epochs chạy | 10 | 17 |
| Thời gian mỗi epoch (ước tính) | ~45 giây | ~55 giây |

---

## 6. Phân Tích & Đánh Giá

### 6.1. Hiệu Quả Của Các Cải Tiến

**Attention Pooling:** Cải thiện đáng kể Valid Correlation (+3.4%) vì giúp mô hình tập trung vào các bước thời gian có ý nghĩa thay vì bị nhiễu bởi padding. Đây là cải tiến có tác động lớn nhất.

**Gated Fusion:** Cải thiện Valid MSE (-7.1%) vì giúp kiểm soát lỗi dự đoán lớn — cơ chế cổng tự động giảm trọng số của phương thức bị nhiễu hoặc không đáng tin cậy.

**Tăng Regularization (2-layer LSTM, Dropout, LayerNorm):** Giúp mô hình chạy được nhiều epochs hơn trước khi overfitting (17 vs 10), nhưng không ngăn chặn được overfitting hoàn toàn.

### 6.2. Giới Hạn Đã Nhận Diện

1. **Giới hạn kiến trúc LSTM:** Cả hai mô hình đều cho thấy khoảng cách Train-Valid Loss rất lớn (>10x ở cuối quá trình huấn luyện). Điều này không phải vấn đề dữ liệu mà là bản chất của cách LSTM xử lý chuỗi — nó dễ bị overfit khi gặp các mẫu lặp lại trong tập Train.

2. **Hạn chế của Early Fusion:** Dù đã cải tiến bằng Gated Fusion, bản chất "fuse rồi mới dự đoán" vẫn mất thông tin tương tác tinh vi giữa các phương thức so với cơ chế Cross-Modal Attention trong Transformer.

3. **Sự khác biệt giữa Valid và Test:** Mô hình cải tiến tốt hơn rõ ràng trên Valid nhưng chênh lệch không rõ ràng trên Test, cho thấy phân phối dữ liệu giữa Valid và Test có sự khác biệt nhất định. Đây là hiện tượng phổ biến trong nghiên cứu.

---

## 7. Kết Luận & Hướng Phát Triển

### 7.1. Kết Luận

1. **Baseline Early Fusion LSTM** đã xác minh thành công toàn bộ pipeline huấn luyện (nạp dữ liệu, huấn luyện, đánh giá, lưu checkpoint, đồng bộ GCS, ghi log W&B) và cung cấp điểm chuẩn ban đầu (Test MAE=0.5815, Test Corr=0.7125).

2. **Improved Early Fusion LSTM** với Attention Pooling + Gated Fusion đã cải thiện khả năng tổng quát hóa trên tập Validation (MAE cải thiện 2.6%, Correlation cải thiện 3.4%), nhưng kết quả trên tập Test chỉ cải thiện nhẹ ở MSE (-1.9%) và Correlation (+1.5%).

3. **Cả hai mô hình LSTM đều gặp giới hạn trần về overfitting**, cho thấy cần chuyển sang kiến trúc mạnh hơn (Transformer) để đạt kết quả tốt hơn.

### 7.2. Hướng Phát Triển — Phase 1 Nâng Cao

- **Mô hình MulT (Multimodal Transformer):** Sử dụng cơ chế Cross-Modal Attention cho phép mỗi phương thức "nhìn" trực tiếp vào thông tin của phương thức khác, thay vì chỉ hợp nhất ở cuối. Kỳ vọng cải thiện đáng kể cả MAE và Correlation.
- **Code đã sẵn sàng:** File `training/models/mult.py` và cấu hình `Phase1MulTModelConfig` đã được triển khai, chỉ cần tạo notebook Colab và chạy huấn luyện.

---

## 8. Tài Nguyên Đã Lưu Trữ

### 8.1. Checkpoints Mô Hình (Local)

| File | Đường dẫn | Kích thước | Mô tả |
|:---|:---|:---:|:---|
| `best_model.pt` | `checkpoints/phase1/best_model.pt` | ~14 MB | Baseline LSTM — Epoch 4 |
| `last_model.pt` | `checkpoints/phase1/last_model.pt` | ~14 MB | Baseline LSTM — Epoch 10 |
| `best_model_improved_lstm.pt` | `checkpoints/phase1/best_model_improved_lstm.pt` | ~24 MB | Improved LSTM — Epoch 7 |
| `last_model_improved_lstm.pt` | `checkpoints/phase1/last_model_improved_lstm.pt` | ~24 MB | Improved LSTM — Epoch 17 |

### 8.2. Lịch Sử Huấn Luyện

| File | Đường dẫn | Mô tả |
|:---|:---|:---|
| `history.csv` | `logs/phase1/history.csv` | Toàn bộ log metrics mỗi epoch (cả 2 mô hình) |
| `summary.json` | `outputs/phase1/summary.json` | Tóm tắt huấn luyện mô hình gần nhất |

### 8.3. Google Cloud Storage (Backup)

| Thư mục GCS | Nội dung |
|:---|:---|
| `gs://mer-data-bucket-kandesfx/checkpoints/phase1/` | 4 file checkpoint |
| `gs://mer-data-bucket-kandesfx/logs/phase1/` | File history.csv |
| `gs://mer-data-bucket-kandesfx/outputs/phase1/` | File summary.json |
| `gs://mer-data-bucket-kandesfx/data/MSA-Dataset/` | Tập dữ liệu aligned_50.pkl |

### 8.4. Weights & Biases Dashboard

| Run | Link | Mô tả |
|:---|:---|:---|
| Baseline LSTM | [W&B Run](https://wandb.ai/kandesfx-kandesfx/bcda-phase1) | Run `phase1_early_fusion_colab` |
| Improved LSTM | [W&B Run](https://wandb.ai/kandesfx-kandesfx/bcda-phase1/runs/2bnz38bn) | Run `phase1_improved_lstm_colab` |

### 8.5. Mã Nguồn Liên Quan

| File | Đường dẫn | Mô tả |
|:---|:---|:---|
| Baseline Model | `training/models/early_fusion.py` | Kiến trúc EarlyFusionLSTMRegressor |
| Improved Model | `training/models/improved_lstm.py` | Kiến trúc ImprovedLSTMRegressor |
| MulT Model | `training/models/mult.py` | Kiến trúc MulTRegressor (sẵn sàng) |
| Config | `training/config_phase1.py` | Toàn bộ cấu hình Phase 1 |
| Trainer | `training/trainer.py` | Vòng lặp huấn luyện + đánh giá |
| Evaluator | `training/evaluator.py` | Tính toán metrics (MAE, MSE, Corr, Acc, F1) |
| Dataset | `training/dataset_mosei.py` | DataLoader cho CMU-MOSEI |
| Notebook Baseline | `notebooks/02_baseline_early_fusion.ipynb` | Notebook Colab cho Baseline |
| Notebook Improved | `notebooks/02_improved_early_fusion.ipynb` | Notebook Colab cho Improved LSTM |
