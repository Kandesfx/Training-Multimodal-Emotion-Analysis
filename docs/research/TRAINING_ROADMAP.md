# 🚀 Chiến Lược Huấn Luyện & Lộ Trình Triển Khai Mô Hình Đa Phương Thức

## Đề Tài 17: Hệ Thống Phân Tích Cảm Xúc Đa Phương Thức với CMU-MOSEI

Tài liệu này trình bày chi tiết phương hướng huấn luyện mô hình, cách điều chỉnh tool Emotion Data Studio (EDS) để xuất dữ liệu chuẩn cho Fine-tuning, và phương pháp căn chỉnh dữ liệu tiếng Việt trong tương lai.

---

## Mục Lục

1. [Tổng Quan Hiện Trạng — So Sánh "Đang Có" và "Cần Có"](#1-tổng-quan-hiện-trạng)
2. [Lộ Trình Huấn Luyện 2 Giai Đoạn](#2-lộ-trình-huấn-luyện-2-giai-đoạn)
3. [Phase 1: Pre-training trên CMU-MOSEI](#3-phase-1-pre-training-trên-cmu-mosei)
4. [Phase 2: Fine-tuning trên Dữ Liệu Tiếng Việt](#4-phase-2-fine-tuning-trên-dữ-liệu-tiếng-việt)
5. [Điều Chỉnh EDS Tool — Pipeline Trích Xuất Đặc Trưng](#5-điều-chỉnh-eds-tool)
6. [Phương Pháp Căn Chỉnh Dữ Liệu Tiếng Việt (Word-level Alignment)](#6-phương-pháp-căn-chỉnh-dữ-liệu-tiếng-việt)
7. [Định Dạng Xuất File .pkl Chuẩn MMSA](#7-định-dạng-xuất-file-pkl-chuẩn-mmsa)
8. [Chiến Lược Nhãn Sentiment Score cho Dữ Liệu Việt](#8-chiến-lược-nhãn-sentiment-score)
9. [Tổng Kết & Bước Tiếp Theo](#9-tổng-kết)

---

## 1. Tổng Quan Hiện Trạng

### 1.1. Dữ Liệu Đã Có Sẵn (CMU-MOSEI — `aligned_50.pkl`)

Từ kết quả chẩn đoán thực tế (xem [DATASET_PREPARATION.md](file:///d:/Hai/study/DeepLerning/BCDA/docs/research/DATASET_PREPARATION.md)), tập dữ liệu CMU-MOSEI đã tải về chứa **22,856 mẫu hội thoại** với đặc trưng đã được trích xuất sẵn:

| Phương thức | Khóa trong `.pkl` | Kích thước mỗi mẫu | Công cụ trích xuất | Ý nghĩa |
|:---|:---|:---|:---|:---|
| **Văn bản** | `'text'` | `(50, 768)` float32 | BERT-base-uncased | Vector ngữ nghĩa 768 chiều cho mỗi từ, đã padding về 50 bước |
| **Âm thanh** | `'audio'` | `(50, 74)` float64 | COVAREP | Vector đặc trưng âm sắc 74 chiều, đã mean-pool về 50 bước |
| **Hình ảnh** | `'vision'` | `(50, 35)` float64 | FACET | 35 Action Units cường độ cơ mặt, đã mean-pool về 50 bước |
| **Nhãn** | `'regression_labels'` | `(1,)` float64 | Human annotation | Điểm sentiment liên tục [-3.0, +3.0] |

**Điểm then chốt:** Đây đã là **đặc trưng cấp cao** (high-level features), KHÔNG phải dữ liệu thô. Tức là:
- `'text'` KHÔNG phải chuỗi ký tự → mà là vector embedding 768 chiều đã qua BERT.
- `'audio'` KHÔNG phải sóng âm thanh `.wav` → mà là vector đặc trưng 74 chiều đã qua COVAREP.
- `'vision'` KHÔNG phải ảnh pixel khuôn mặt → mà là vector 35 Action Units đã qua FACET.

### 1.2. Đầu Ra Hiện Tại của EDS Tool

Tool Emotion Data Studio (EDS) tại `tools/emotion-data-studio/` hiện tại có pipeline xử lý 1 video clip như sau:

```
Video .mp4
  │
  ├──► [FaceExtractor]     → Ảnh khuôn mặt crop .jpg (24 frames mẫu)
  │                          → detections.json (bounding boxes + track IDs)
  │
  ├──► [AudioExtractor]    → File .wav (16kHz, mono)
  │                          → 40 hệ số MFCC trung bình (1 vector duy nhất cho toàn clip)
  │                          → audio_clarity, tempo, zero_crossing_rate
  │
  ├──► [SpeechTranscriber] → Chuỗi transcript tiếng Việt (string)
  │                          → Word segments với timestamps
  │
  └──► [EmotionAnalyzer]   → Nhãn cảm xúc rời rạc 7 lớp (happy/sad/angry/...)
                              → Confidence score, agreement
```

### 1.3. Bảng So Sánh "Đang Có" vs "Cần Có"

| Thành phần | EDS hiện tại (Đang có) | MMSA cần (Cần có) | Khoảng cách |
|:---|:---|:---|:---|
| **Visual** | Ảnh `.jpg` crop khuôn mặt (pixel thô) | Vector 35 Action Units per-frame | ❌ **Thiếu hoàn toàn** — cần thêm module trích xuất AUs |
| **Audio** | 1 vector MFCC 40 chiều trung bình toàn clip | Chuỗi vector 74 chiều @100Hz frame-by-frame | ❌ **Thiếu hoàn toàn** — cần trích xuất frame-level |
| **Text** | Chuỗi ký tự transcript tiếng Việt | Vector embedding 768 chiều per-word | ❌ **Thiếu hoàn toàn** — cần chạy qua PhoBERT |
| **Nhãn** | 7 lớp cảm xúc rời rạc (happy, sad, ...) | Điểm sentiment liên tục [-3.0, +3.0] | ⚠️ **Cần bổ sung** — thêm slider sentiment |
| **Căn chỉnh** | Không có (mỗi kênh xử lý độc lập) | Word-level alignment (3 kênh cùng trục) | ❌ **Thiếu hoàn toàn** — cần alignment engine |
| **Xuất file** | Lưu SQLite database | File `.pkl` dictionary chuẩn MMSA | ❌ **Thiếu hoàn toàn** — cần MMSA exporter |

---

## 2. Lộ Trình Huấn Luyện 2 Giai Đoạn

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                      LỘ TRÌNH TỔNG THỂ                                │
  │                                                                         │
  │  [PHASE 1]                              [PHASE 2]                      │
  │  Pre-training trên CMU-MOSEI   ──────►  Fine-tuning trên dữ liệu Việt │
  │                                                                         │
  │  Dữ liệu: aligned_50.pkl               Dữ liệu: vietnamese_50.pkl    │
  │  (22,856 mẫu tiếng Anh)                (N mẫu tiếng Việt từ EDS)     │
  │                                                                         │
  │  Nhãn: regression_labels [-3,+3]        Nhãn: sentiment_score [-3,+3] │
  │  (có sẵn)                               (gán bằng EDS tool)           │
  │                                                                         │
  │  Kết quả: Mô hình đã học                Kết quả: Mô hình tinh chỉnh  │
  │  cách kết hợp 3 phương thức             cho ngữ cảnh tiếng Việt       │
  │  → Lưu checkpoint .pt                   → Lưu checkpoint cuối .pt     │
  └─────────────────────────────────────────────────────────────────────────┘
```

**Tại sao chia 2 giai đoạn?**
- **Phase 1** sử dụng tập dữ liệu lớn CMU-MOSEI (22,856 mẫu) giúp mô hình học được **cách kết hợp** thông tin đa phương thức hiệu quả. Đây là bước "học kỹ năng nền tảng".
- **Phase 2** sử dụng tập dữ liệu nhỏ hơn bằng tiếng Việt (cào từ EDS) để mô hình **tinh chỉnh** cho ngữ cảnh văn hóa, giọng điệu và biểu cảm của người Việt. Đây là bước "chuyên môn hóa".

---

## 3. Phase 1: Pre-training trên CMU-MOSEI

### 3.1. Tại Sao Dùng LSTM Mà Không Dùng CNN?

Đây là câu hỏi quan trọng nhất cần làm rõ. Trong file `aligned_50.pkl`, dữ liệu đã ở dạng **vector đặc trưng cấp cao** (không phải dữ liệu thô):

```
                Dữ liệu thô (RAW)                    Đặc trưng đã trích xuất (FEATURES)
                ─────────────────                      ────────────────────────────────────
  Video:        Ảnh pixel 224×224×3                    35 số thực (Action Units)
  Audio:        Sóng âm thanh .wav                     74 số thực (COVAREP)
  Text:         Chuỗi ký tự "I hate it"               768 số thực (BERT embedding)
                        │                                          │
                        │ ← CNN/BERT cần thiết ở đây               │ ← LSTM cần thiết ở đây
                        │   (trích xuất đặc trưng                  │   (mô hình hóa biến thiên
                        │    từ dữ liệu thô)                      │    theo thời gian)
```

**Giải thích:**
- **CNN** cần thiết khi đầu vào là dữ liệu thô (ảnh pixel, sóng âm) → để **tìm các pattern không gian** (cạnh, góc, texture). Nhưng FACET/COVAREP/BERT đã làm việc này rồi.
- **LSTM** cần thiết để **mô hình hóa chuỗi thời gian** dài 50 bước → để học sự biến thiên của biểu cảm, giọng điệu, ngữ nghĩa qua thời gian trong câu thoại.

Nói cách khác, phần trích xuất đặc trưng (Feature Extraction) đã được hoàn thành bởi BERT/COVAREP/FACET. Mô hình của chúng ta chỉ cần tập trung vào phần **Temporal Modeling** (mô hình hóa thời gian) và **Fusion** (kết hợp đa phương thức).

### 3.2. Kiến Trúc Mô Hình Phase 1

```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  text (50, 768)  ──► LSTM(input=768, hidden=128, bidirectional)        │
  │                      ──► Lấy hidden state cuối cùng ──► h_text (256)   │
  │                                                                │        │
  │  audio (50, 74)  ──► LSTM(input=74,  hidden=64,  bidirectional)        │
  │                      ──► Lấy hidden state cuối cùng ──► h_audio (128)  │
  │                                                                │        │
  │  vision (50, 35) ──► LSTM(input=35,  hidden=64,  bidirectional)        │
  │                      ──► Lấy hidden state cuối cùng ──► h_video (128)  │
  │                                                                │        │
  │                                                                ▼        │
  │                                                    ┌─────────────────┐  │
  │                                                    │ CONCATENATION   │  │
  │                                                    │ (256+128+128    │  │
  │                                                    │  = 512 chiều)   │  │
  │                                                    └────────┬────────┘  │
  │                                                             │           │
  │                                                             ▼           │
  │                                               ┌──────────────────────┐  │
  │                                               │   FUSION NETWORK     │  │
  │                                               │                      │  │
  │                                               │   Linear(512 → 256)  │  │
  │                                               │   BatchNorm + ReLU   │  │
  │                                               │   Dropout(0.3)       │  │
  │                                               │                      │  │
  │                                               │   Linear(256 → 128)  │  │
  │                                               │   ReLU               │  │
  │                                               │   Dropout(0.2)       │  │
  │                                               │                      │  │
  │                                               │   Linear(128 → 1)    │  │ ← Đầu ra Hồi quy
  │                                               └──────────────────────┘  │
  │                                                             │           │
  │                                                             ▼           │
  │                                                   ŷ ∈ [-3.0, +3.0]     │
  │                                                  (Sentiment Score)      │
  └─────────────────────────────────────────────────────────────────────────┘
```

### 3.3. Giải Thích Chi Tiết Từng Thành Phần

#### A. LSTM Encoder (3 nhánh riêng biệt)
- **Mục tiêu:** Nhận chuỗi 50 vector đặc trưng theo thời gian → xuất ra 1 vector cô đọng đại diện cho toàn bộ chuỗi.
- **Bi-directional:** Đọc chuỗi theo cả chiều xuôi (từ đầu → cuối câu) và chiều ngược (từ cuối → đầu câu) để nắm bắt ngữ cảnh toàn diện hơn.
- **Kích thước ẩn (hidden size):**
  - Text: 128 → output 256 (bi-directional nên nhân đôi: 128 × 2)
  - Audio: 64 → output 128
  - Vision: 64 → output 128
  - **Tổng:** 256 + 128 + 128 = **512 chiều** nạp vào Fusion.

#### B. Fusion Network (Mạng kết hợp đặc trưng)
- **Phương pháp:** Early Fusion (Nối trực tiếp 3 vector → Fully Connected layers).
- **Lý do chọn Early Fusion:** Đơn giản, hiệu quả, dễ debug, phù hợp cho Phase 1 khi mục tiêu chính là verify pipeline hoạt động đúng.
- **Các lớp nối:**
  - `512 → 256`: Thu nhỏ và trộn thông tin đa phương thức.
  - `256 → 128`: Nén tiếp, học tương tác phi tuyến giữa các phương thức.
  - `128 → 1`: Đầu ra hồi quy — 1 số thực duy nhất đại diện sentiment score.

#### C. Hàm Loss và Đánh Giá
- **Loss function:** `MSELoss` — Tính sai số bình phương trung bình giữa $\hat{y}$ (dự đoán) và $y$ (nhãn thực):
  $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$$
- **Metrics đánh giá** (theo chuẩn MMSA):
  - **MAE** (Mean Absolute Error): Sai số tuyệt đối trung bình.
  - **Corr** (Pearson Correlation): Hệ số tương quan tuyến tính giữa dự đoán và nhãn.
  - **Acc-2** (Has0): Chuyển đầu ra hồi quy thành nhị phân (≥0 = Positive, <0 = Negative), tính accuracy.
  - **Acc-7**: Làm tròn đầu ra hồi quy về [-3, -2, -1, 0, 1, 2, 3], tính accuracy 7 lớp.
  - **F1-score**: F1 có trọng số cho phân loại nhị phân.

### 3.4. Thông Số Huấn Luyện Đề Xuất

| Thông số | Giá trị | Ghi chú |
|:---|:---|:---|
| **Batch size** | 32 hoặc 64 | Tùy thuộc VRAM GPU |
| **Learning rate** | 1e-3 | Dùng Adam optimizer |
| **LR scheduler** | ReduceLROnPlateau | Giảm LR khi val_loss không giảm sau 3 epoch |
| **Epochs** | 50 (early stopping patience = 8) | Dừng sớm nếu val_loss không cải thiện |
| **Dropout** | 0.3 (fusion layer 1), 0.2 (fusion layer 2) | Chống overfitting |
| **Gradient clipping** | max_norm = 1.0 | Chống gradient explosion trong LSTM |
| **Weight decay** | 1e-4 | L2 regularization |
| **Tập Train/Valid/Test** | 16,326 / 1,871 / 4,659 | Đã chia sẵn trong file `.pkl` |

### 3.5. Xử Lý Đặc Biệt Trong Dữ Liệu

Từ mã nguồn `data_loader.py` của MMSA, có một bước tiền xử lý quan trọng:
```python
# Thay thế giá trị -inf trong audio features bằng 0
self.audio[self.audio == -np.inf] = 0
```
Lý do: Bộ trích xuất COVAREP đôi khi trả về giá trị `-inf` cho các khung hình mà nó không tính được đặc trưng (ví dụ: đoạn im lặng hoàn toàn). Nếu không xử lý, `-inf` sẽ gây lỗi `NaN` trong quá trình tính toán gradient.

---

## 4. Phase 2: Fine-tuning trên Dữ Liệu Tiếng Việt

### 4.1. Nguyên Lý

Sau Phase 1, mô hình đã học được:
- Cách LSTM mô hình hóa biến thiên cảm xúc theo thời gian.
- Cách Fusion Network kết hợp thông tin từ 3 kênh để đưa ra dự đoán sentiment.
- Các trọng số (weights) trong fusion layers phản ánh "công thức kết hợp" tối ưu cho tiếng Anh.

Phase 2 sẽ **giữ nguyên kiến trúc**, nạp checkpoint từ Phase 1, nhưng **thay thế dữ liệu** bằng tập dữ liệu tiếng Việt (cào từ EDS):
- Nhánh Text: Thay BERT tiếng Anh (768 chiều) → PhoBERT tiếng Việt (768 chiều). Số chiều giống nhau nên kiến trúc LSTM không cần thay đổi.
- Nhánh Audio/Vision: Giữ nguyên vì biểu cảm khuôn mặt và tông giọng là phổ quát toàn cầu (xem lý thuyết Paul Ekman trong [FINE_TUNING_STRATEGY.md](file:///d:/Hai/study/DeepLerning/BCDA/docs/research/FINE_TUNING_STRATEGY.md)).

### 4.2. Chiến Lược Đóng Băng

```
  Phase 2 — Trạng thái đóng băng/mở khóa:

  text_lstm:    ❄️ ĐÓng băng (đã học cách mô hình hóa chuỗi văn bản, giữ nguyên)
  audio_lstm:   ❄️ Đóng băng (đã học cách mô hình hóa chuỗi âm thanh, giữ nguyên)
  vision_lstm:  ❄️ Đóng băng (đã học cách mô hình hóa chuỗi hình ảnh, giữ nguyên)

  fusion_layer1: 🔓 Mở khóa (cần tinh chỉnh "công thức kết hợp" cho ngữ cảnh Việt)
  fusion_layer2: 🔓 Mở khóa
  output_layer:  🔓 Mở khóa

  Learning Rate: 1e-4 (nhỏ hơn Phase 1 gấp 10 lần để tránh phá vỡ kiến thức đã học)
```

### 4.3. Dự Kiến Kích Thước Dữ Liệu Việt Cần Thiết

| Kịch bản | Số mẫu | Hiệu quả dự kiến |
|:---|:---|:---|
| Tối thiểu | 500-1000 clips | Fusion layers học được pattern cơ bản của tiếng Việt |
| Khuyên dùng | 2000-5000 clips | Đủ đa dạng ngữ cảnh, giọng vùng miền, biểu cảm |
| Lý tưởng | 5000-10000 clips | Bao phủ tốt 7 loại cảm xúc, nhiều chủ đề phim/talkshow |

---

## 5. Điều Chỉnh EDS Tool — Pipeline Trích Xuất Đặc Trưng

Đây là phần quan trọng nhất cần bổ sung vào EDS. Hiện tại EDS chỉ có pipeline **annotation** (gán nhãn), cần thêm pipeline **feature extraction** (trích xuất đặc trưng cấp khung hình).

### 5.1. Module Mới #1: Audio Feature Extractor (74 chiều frame-level)

**Vị trí:** `backend/services/feature_extractors/audio_feature_extractor.py`

**Vấn đề hiện tại:** Module `audio_extractor.py` hiện tại tính 40 hệ số MFCC trung bình cho toàn bộ clip → chỉ ra 1 vector duy nhất. Mô hình MMSA cần chuỗi vector theo trục thời gian (**frame-level**) tại tần số 100Hz.

**Giải pháp — Tạo vector 74 chiều tương đương COVAREP bằng Librosa:**

COVAREP gốc trích xuất 74 đặc trưng âm sắc. Ta có thể tái tạo tương đương bằng Librosa (Python):

| Nhóm đặc trưng | Số chiều | Công cụ Librosa | Mô tả |
|:---|:---|:---|:---|
| MFCC (static) | 13 | `librosa.feature.mfcc(n_mfcc=13)` | Hệ số Mel cơ bản |
| MFCC delta (vận tốc) | 13 | `librosa.feature.delta(mfcc)` | Tốc độ thay đổi MFCC |
| MFCC delta-delta (gia tốc) | 13 | `librosa.feature.delta(mfcc, order=2)` | Gia tốc thay đổi MFCC |
| Chroma (sắc độ tần số) | 12 | `librosa.feature.chroma_stft()` | Phân bố năng lượng 12 nốt nhạc |
| Spectral Contrast | 7 | `librosa.feature.spectral_contrast()` | Tương phản phổ tần |
| ZCR (Zero Crossing Rate) | 1 | `librosa.feature.zero_crossing_rate()` | Tần suất đổi dấu sóng |
| RMS Energy | 1 | `librosa.feature.rms()` | Năng lượng trung bình bình phương gốc |
| Spectral Centroid | 1 | `librosa.feature.spectral_centroid()` | Trọng tâm phổ tần |
| Spectral Bandwidth | 1 | `librosa.feature.spectral_bandwidth()` | Độ rộng băng tần |
| Spectral Rolloff | 1 | `librosa.feature.spectral_rolloff()` | Tần số cuộn phổ |
| Spectral Flatness | 1 | `librosa.feature.spectral_flatness()` | Độ phẳng phổ (tín hiệu tonal vs noise) |
| F0 / Pitch (tần số cơ bản) | 1 | `librosa.pyin()` | Cao độ giọng nói |
| Voiced/Unvoiced Flag | 1 | Từ kết quả `pyin` | Cờ đánh dấu có giọng nói hay không |
| Harmonic-to-Noise Ratio | 1 | `librosa.effects.hpss()` | Tỉ lệ hài trên nhiễu |
| Tonnetz | 6 | `librosa.feature.tonnetz()` | Biểu diễn harmonic tonal centroid |
| **TỔNG** | **74** | | |

**Cài đặt kỹ thuật quan trọng:**
- Tần số lấy mẫu: `sr = 16000` Hz
- Hop length: `hop_length = 160` samples → tạo ra 100 khung hình mỗi giây (100Hz), khớp với COVAREP
- Window length: `n_fft = 512` (32ms @ 16kHz)
- Output shape: `(T_audio, 74)` với `T_audio = duration_sec × 100`

### 5.2. Module Mới #2: Visual Feature Extractor (35 Action Units frame-level)

**Vị trí:** `backend/services/feature_extractors/visual_feature_extractor.py`

**Vấn đề hiện tại:** Module `face_extractor.py` chỉ crop ảnh khuôn mặt lưu dạng `.jpg`. Mô hình MMSA cần vector 35 Action Units cho mỗi khung hình video.

**Giải pháp — 2 phương án:**

#### Phương án A: OpenFace 2.0 (Chính xác cao)

```
Cài đặt:
  1. Tải OpenFace binary từ https://github.com/TadasBaltrusaitis/OpenFace
  2. Đặt vào thư mục tools/emotion-data-studio/bin/OpenFace/

Chạy:
  FeatureExtraction.exe -f clip.mp4 -aus -out_dir output_dir/

Kết quả:
  → File CSV chứa 35 cột AU intensities cho mỗi frame:
    AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r,
    AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r,
    AU28_r, AU45_r + 17 cột AU presence (binary)
  → Parse CSV → numpy array shape (T_video, 35)
```

#### Phương án B: Py-Feat / MediaPipe (Dễ cài đặt)

```
Cài đặt:
  pip install py-feat

Chạy:
  from feat import Detector
  detector = Detector(face_model="retinaface", au_model="xgb")
  result = detector.detect_image(frame)
  aus = result.aus  # 20 Action Units

Hạn chế:
  → Py-Feat chỉ trích xuất ~20 AUs (không đủ 35)
  → Cần padding thêm 15 chiều = 0 hoặc sử dụng kết hợp
     với MediaPipe Face Mesh để bổ sung landmark features
```

**Khuyến nghị:** Ưu tiên **Phương án A (OpenFace)** nếu môi trường cho phép, vì kết quả chính xác hơn và tương thích trực tiếp với 35 chiều của FACET trong CMU-MOSEI.

### 5.3. Module Mới #3: Text Feature Extractor (PhoBERT 768 chiều per-word)

**Vị trí:** `backend/services/feature_extractors/text_feature_extractor.py`

**Vấn đề hiện tại:** Module `transcriber.py` xuất ra chuỗi ký tự transcript. Mô hình MMSA cần vector embedding 768 chiều cho từng từ.

**Quy trình xử lý:**

```
Input: Transcript tiếng Việt + Whisper word timestamps

Bước 1: Tokenize bằng PhoBERT Tokenizer
  tokens = tokenizer.tokenize("Tôi rất ghét bộ phim này")
  → ["Tôi", "rất", "ghét", "bộ_phim", "này"]

Bước 2: Chạy qua PhoBERT model (đóng băng trọng số)
  with torch.no_grad():
      outputs = phobert(input_ids, attention_mask)
      embeddings = outputs.last_hidden_state  # (1, num_tokens, 768)

Bước 3: Xử lý subword tokens
  PhoBERT có thể chia 1 từ thành nhiều subword tokens.
  Ví dụ: "phim" → ["ph", "##im"]
  → Lấy trung bình vector của các subword tokens cùng 1 từ gốc
  → Output: (num_words, 768) — 1 vector 768 chiều cho mỗi từ tiếng Việt

Output: Ma trận (T_text, 768) + word_timestamps [(word, t_start, t_end), ...]
```

**Điểm tương thích:** PhoBERT có kích thước đầu ra **768 chiều**, giống hệt BERT-base-uncased trong CMU-MOSEI. Điều này đảm bảo kiến trúc LSTM text encoder của mô hình (input_dim=768) hoạt động mà **không cần thay đổi bất kỳ số chiều nào**.

### 5.4. Sơ Đồ Pipeline EDS Sau Khi Nâng Cấp

```
  Video .mp4 từ EDS                    Pipeline Annotation    Pipeline Feature Extraction
  ─────────────────                    (Đã có - giữ nguyên)    (MỚI - bổ sung)
        │                                    │                         │
        ├──► FaceExtractor ────────► Ảnh .jpg + detections    ──────────┘
        │         │                                                     │
        │         └──────────────────────────────────────────► Visual Feature Extractor
        │                                                      (OpenFace → 35 AUs @30Hz)
        │                                                              │
        ├──► AudioExtractor ───────► .wav + MFCC trung bình   ──────────┘
        │         │                                                     │
        │         └──────────────────────────────────────────► Audio Feature Extractor
        │                                                      (Librosa → 74 dim @100Hz)
        │                                                              │
        ├──► SpeechTranscriber ────► transcript + timestamps  ──────────┘
        │         │                                                     │
        │         └──────────────────────────────────────────► Text Feature Extractor
        │                                                      (PhoBERT → 768 dim per-word)
        │                                                              │
        │                                                              ▼
        │                                                    ┌──────────────────┐
        │                                                    │ Alignment Engine │
        │                                                    │ (Word-level      │
        │                                                    │  Mean Pooling)   │
        │                                                    └────────┬─────────┘
        │                                                             │
        │                                                             ▼
        │                                                    ┌──────────────────┐
        │                                                    │ MMSA Exporter    │
        └──► EmotionAnalyzer ─────► Nhãn cảm xúc            │ (.pkl output)    │
                                    + sentiment score ──────►│                  │
                                                             └──────────────────┘
```

---

## 6. Phương Pháp Căn Chỉnh Dữ Liệu Tiếng Việt (Word-level Alignment)

### 6.1. Tại Sao Cần Căn Chỉnh?

Ba luồng đặc trưng có tần số lấy mẫu khác nhau hoàn toàn:

| Phương thức | Tần số gốc | Ví dụ: Clip 5 giây | Tổng số vector |
|:---|:---|:---|:---|
| Text | Theo từ đơn | "Tôi rất ghét bộ phim này" | 6 vector |
| Audio | 100Hz | 5s × 100 = 500 | 500 vector |
| Vision | 30Hz | 5s × 30 = 150 | 150 vector |

Nếu không căn chỉnh, mô hình LSTM sẽ nhận chuỗi có độ dài khác nhau cho 3 nhánh → không thể nối (concatenate) các hidden state đúng thời điểm.

### 6.2. Thuật Toán Căn Chỉnh Chi Tiết

#### Bước 1: Xác Định Mốc Thời Gian Từng Từ (Word Timestamps)

Trong CMU-MOSEI gốc, các tác giả sử dụng **Forced Aligner** (P2FA hoặc Gentle) để tìm thời gian phát âm của từng từ. Đối với tiếng Việt, ta tận dụng **Whisper** có sẵn trong EDS:

```python
# Whisper đã tích hợp sẵn word-level timestamps
result = whisper_model.transcribe(
    audio_path,
    language="vi",
    word_timestamps=True  # ← Kích hoạt timestamps cấp từ
)

# Kết quả:
# result['segments'][0]['words'] = [
#     {'word': 'Tôi',   'start': 0.12, 'end': 0.35},
#     {'word': 'rất',   'start': 0.36, 'end': 0.52},
#     {'word': 'ghét',  'start': 0.55, 'end': 0.88},
#     {'word': 'bộ',    'start': 0.90, 'end': 1.05},
#     {'word': 'phim',  'start': 1.06, 'end': 1.28},
#     {'word': 'này',   'start': 1.30, 'end': 1.55},
# ]
```

#### Bước 2: Cắt & Gộp Đặc Trưng Audio/Video Theo Mốc Thời Gian

```
  Ví dụ minh họa cho từ "ghét" (phát âm từ 0.55s đến 0.88s):

  Audio Features (100Hz):
  ┌────────────────────────────────────────────────────────┐
  │ frame 0  frame 1  ...  frame 54  frame 55  ...  frame 87  frame 88  ...  │
  │ (0.00s)  (0.01s)      (0.54s)   (0.55s)       (0.87s)   (0.88s)        │
  │                                  ├──────────────────────┤                 │
  │                                  │  33 vectors (74 dim) │                 │
  │                                  │  thuộc về từ "ghét"  │                 │
  │                                  └──────────┬───────────┘                 │
  └─────────────────────────────────────────────│─────────────────────────────┘
                                                │
                                                ▼ Mean Pooling (theo axis=0)
                                       ┌────────────────┐
                                       │ 1 vector 74 dim│ ← Đại diện cho từ "ghét"
                                       │ = mean(33 vecs)│    trong không gian Audio
                                       └────────────────┘

  Video Features (30Hz):
  ┌────────────────────────────────────────────────────────┐
  │ frame 0  ...  frame 16  frame 17  ...  frame 26  ...  │
  │ (0.00s)      (0.53s)   (0.57s)       (0.87s)         │
  │                         ├──────────────────┤           │
  │                         │ 10 vectors       │           │
  │                         │ (35 dim)         │           │
  │                         └────────┬─────────┘           │
  └──────────────────────────────────│─────────────────────┘
                                     │
                                     ▼ Mean Pooling (theo axis=0)
                                ┌────────────────┐
                                │ 1 vector 35 dim│ ← Đại diện cho từ "ghét"
                                │ = mean(10 vecs)│    trong không gian Video
                                └────────────────┘
```

#### Bước 3: Lặp Lại Cho Tất Cả Từ → Thu Được Ma Trận Đã Căn Chỉnh

```
  Từ:         "Tôi"    "rất"    "ghét"    "bộ"     "phim"    "này"
  ──────────────────────────────────────────────────────────────────
  Text:       [768]    [768]    [768]     [768]    [768]     [768]   ← Giữ nguyên từ PhoBERT
  Audio:      [74]     [74]     [74]      [74]     [74]      [74]   ← Mean-pooled per word
  Vision:     [35]     [35]     [35]      [35]     [35]      [35]   ← Mean-pooled per word

  → Kết quả: 3 ma trận đồng bộ hoàn hảo theo trục thời gian:
     text_aligned:   (6, 768)
     audio_aligned:  (6, 74)
     vision_aligned: (6, 35)
```

#### Bước 4: Padding/Truncation Về Chiều Dài Cố Định 50

```python
MAX_SEQ_LEN = 50

def pad_or_truncate(features, max_len=50):
    """Đệm zero hoặc cắt bớt để mọi chuỗi có chiều dài = max_len."""
    seq_len, feat_dim = features.shape
    if seq_len >= max_len:
        return features[:max_len]  # Cắt bỏ phần dư
    else:
        # Đệm zero vectors ở cuối
        padding = np.zeros((max_len - seq_len, feat_dim))
        return np.concatenate([features, padding], axis=0)

# Áp dụng:
text_final   = pad_or_truncate(text_aligned)    # (50, 768)
audio_final  = pad_or_truncate(audio_aligned)   # (50, 74)
vision_final = pad_or_truncate(vision_aligned)  # (50, 35)
```

### 6.3. Xử Lý Các Trường Hợp Đặc Biệt

| Tình huống | Xử lý |
|:---|:---|
| Từ quá ngắn (< 10ms), không có frame audio/video nào | Sao chép vector của từ liền kề gần nhất |
| Đoạn im lặng kéo dài (không phát hiện từ nào) | Bỏ qua — chỉ xử lý các khoảng thời gian có từ |
| Whisper không trả về word timestamps | Fallback: chia đều thời gian clip cho số từ trong transcript |
| Clip quá ngắn (< 1 giây) hoặc không có transcript | Loại bỏ khỏi tập training |
| Số từ > 50 | Cắt bỏ các từ cuối, giữ lại 50 từ đầu tiên |

---

## 7. Định Dạng Xuất File `.pkl` Chuẩn MMSA

### 7.1. Cấu Trúc Dictionary Chuẩn

Module MMSA Exporter sẽ xuất file `vietnamese_aligned_50.pkl` với cấu trúc dictionary giống hệt format MMSA:

```python
output_data = {
    'train': {
        'raw_text':             np.array(list_of_transcripts),          # (N_train,)
        'text':                 np.array(text_features, dtype=np.float32),  # (N_train, 50, 768)
        'audio':                np.array(audio_features, dtype=np.float64), # (N_train, 50, 74)
        'vision':               np.array(vision_features, dtype=np.float64),# (N_train, 50, 35)
        'id':                   list_of_clip_ids,                       # List[str]
        'regression_labels':    np.array(sentiment_scores, dtype=np.float64), # (N_train,)
        'classification_labels':np.array(class_labels, dtype=np.float64),     # (N_train,)
        'annotations':          list_of_annotation_strings,             # List[str]
    },
    'valid': { ... },  # Cùng cấu trúc
    'test':  { ... },  # Cùng cấu trúc
}

import pickle
with open('vietnamese_aligned_50.pkl', 'wb') as f:
    pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

### 7.2. Chiến Lược Chia Tập Train/Valid/Test

```
Tổng số clips approved trong EDS database
              │
              ├──► 80% → Train
              ├──► 10% → Valid (Validation / Dev set)
              └──► 10% → Test

Quy tắc chia:
  - Chia theo VIDEO (không phải theo clip) để tránh data leakage
    (tránh clip cùng 1 video xuất hiện ở cả train và test)
  - Xáo trộn ngẫu nhiên (shuffle) danh sách video trước khi chia
  - Đảm bảo phân bố cảm xúc tương đối đồng đều trong cả 3 tập
    (Stratified split theo nhãn sentiment)
```

---

## 8. Chiến Lược Nhãn Sentiment Score cho Dữ Liệu Việt

### 8.1. Vấn Đề

CMU-MOSEI sử dụng nhãn **sentiment liên tục** [-3, +3] do annotator gán trực tiếp bằng thang đo Likert 7 điểm. EDS hiện tại chỉ gán nhãn **cảm xúc rời rạc** 7 lớp (happy, sad, angry, ...).

### 8.2. Hai Phương Án Giải Quyết

#### Phương án A: Mapping cứng (Nhanh, dễ triển khai)

Tự động chuyển đổi nhãn cảm xúc rời rạc → sentiment score:

| Nhãn cảm xúc | Sentiment Score | Phân loại |
|:---|:---|:---|
| `happy` | **+2.0** | Positive |
| `surprise` | **+1.0** | Positive (nhẹ) |
| `neutral` | **0.0** | Neutral |
| `sad` | **-1.0** | Negative (nhẹ) |
| `fear` | **-1.5** | Negative |
| `angry` | **-2.0** | Negative (mạnh) |
| `disgust` | **-2.5** | Negative (mạnh) |

- **Ưu điểm:** Triển khai ngay được, không cần thay đổi giao diện EDS.
- **Nhược điểm:** Thô, không phân biệt được mức độ (ví dụ: "hơi vui" vs "cực kỳ vui" đều → +2.0).

#### Phương án B: Thanh trượt liên tục (Chính xác, cần sửa UI)

Bổ sung thanh trượt (slider) [-3, +3] trên giao diện duyệt clip EDS:

```
  Trang Duyệt Clip EDS (UI hiện tại):
  ┌───────────────────────────────────────────┐
  │ [Video Player]                            │
  │                                           │
  │ Nhãn AI: happy (87%)                      │
  │ Nhãn Người duyệt: [Dropdown: 7 cảm xúc]  │
  │                                           │
  │ ★ MỚI: Sentiment Score                    │
  │ -3 ━━━━━━━━━━━━━●━━━━━━━━━━━━━ +3         │
  │         Tiêu cực ← → Tích cực             │
  │         Giá trị: +1.5                     │
  │                                           │
  │ [Approve]  [Reject]  [Skip]               │
  └───────────────────────────────────────────┘
```

- **Ưu điểm:** Chính xác, phân biệt mức độ tinh tế, giống cách CMU-MOSEI gán nhãn.
- **Nhược điểm:** Cần sửa giao diện frontend, tốn thêm thời gian annotator.

**Khuyến nghị:** Bắt đầu bằng **Phương án A** (mapping cứng) để có thể training ngay. Sau đó nếu kết quả Phase 2 chưa đạt yêu cầu, nâng cấp lên **Phương án B** để cải thiện chất lượng nhãn.

---

## 9. Tổng Kết & Bước Tiếp Theo

### 9.1. Tóm Tắt Toàn Bộ Lộ Trình

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  BƯỚC 1: Phase 1 Pre-training                                  │
  │  → Viết code Dataset + Model + Training script                 │
  │  → Huấn luyện trên aligned_50.pkl (CMU-MOSEI)                 │
  │  → Đánh giá MAE, Corr, Acc-2 trên tập test                   │
  │  → Lưu checkpoint best_model.pt                                │
  ├─────────────────────────────────────────────────────────────────┤
  │  BƯỚC 2: Nâng cấp EDS Feature Extraction                      │
  │  → Thêm Audio Feature Extractor (74 dim @100Hz)               │
  │  → Thêm Visual Feature Extractor (35 AUs @30Hz)               │
  │  → Thêm Text Feature Extractor (PhoBERT 768 dim)              │
  │  → Thêm Alignment Engine (Word-level Mean Pooling)            │
  │  → Thêm MMSA Exporter (.pkl)                                  │
  ├─────────────────────────────────────────────────────────────────┤
  │  BƯỚC 3: Thu thập dữ liệu Việt bằng EDS nâng cấp             │
  │  → Cào video tiếng Việt (phim, talkshow, phỏng vấn)          │
  │  → Duyệt + gán nhãn sentiment score                          │
  │  → Xuất vietnamese_aligned_50.pkl                              │
  ├─────────────────────────────────────────────────────────────────┤
  │  BƯỚC 4: Phase 2 Fine-tuning                                   │
  │  → Load checkpoint Phase 1                                     │
  │  → Đóng băng LSTM encoders, mở khóa Fusion layers            │
  │  → Fine-tune trên vietnamese_aligned_50.pkl                    │
  │  → Đánh giá kết quả cuối cùng                                │
  └─────────────────────────────────────────────────────────────────┘
```

### 9.2. Thứ Tự Ưu Tiên Khuyến Nghị

1. **Bắt đầu ngay Phase 1** — vì dữ liệu CMU-MOSEI đã có sẵn trên đĩa, không cần chờ đợi gì thêm.
2. **Song song nâng cấp EDS** — trong lúc Phase 1 đang train, có thể bắt đầu code các module feature extractor mới.
3. **Thu thập dữ liệu Việt** — sau khi EDS đã có pipeline feature extraction hoạt động.
4. **Phase 2 Fine-tuning** — cuối cùng, khi cả model checkpoint và dữ liệu Việt đều đã sẵn sàng.

### 9.3. Các Tài Liệu Liên Quan

- [DATASET_PREPARATION.md](file:///d:/Hai/study/DeepLerning/BCDA/docs/research/DATASET_PREPARATION.md) — Chẩn đoán chi tiết cấu trúc dữ liệu CMU-MOSEI trên đĩa.
- [FINE_TUNING_STRATEGY.md](file:///d:/Hai/study/DeepLerning/BCDA/docs/research/FINE_TUNING_STRATEGY.md) — Chiến lược kiến trúc mô hình và lý thuyết Transfer Learning.
- [DATA_COLLECTION_STRATEGY.md](file:///d:/Hai/study/DeepLerning/BCDA/docs/research/DATA_COLLECTION_STRATEGY.md) — Phương pháp thu thập dữ liệu tiếng Việt & tự động hóa.
- [ARCHITECTURE.md](file:///d:/Hai/study/DeepLerning/BCDA/docs/architecture/ARCHITECTURE.md) — Kiến trúc tổng thể hệ thống.

---

## 10. So Sánh Hai Kiến Trúc Mô Hình: Dữ Liệu Thô vs Đặc Trưng Trích Xuất Sẵn

Dự án hiện có 2 hướng đi kiến trúc. Phần này phân tích rõ sự khác biệt để tránh nhầm lẫn.

### 10.1. Kiến Trúc A — Xử Lý Dữ Liệu Thô (config.py hiện tại)

File [training/config.py](file:///d:/Hai/study/DeepLerning/BCDA/training/config.py) hiện tại thiết kế cho trường hợp đầu vào là **dữ liệu thô** (ảnh pixel, sóng âm, văn bản):

```
  Input thô:                     Backbone (Trích xuất)         Temporal           Fusion
  ──────────                     ────────────────────           ─────────          ──────
  Ảnh khuôn mặt 224×224 ──────► ResNet50 (freeze 6 layers) ──► Temporal LSTM ──┐
                                  → feature_dim = 512           → hidden = 256   │
                                                                                  │
  Audio .wav 16kHz ─────────────► CNN 1D [64,128,256] ────────► BiLSTM ────────┼──► Concat → FC → 7 lớp
                                  → 120 dim (40 MFCC×3)         → hidden = 256  │     (512×3=1536)
                                                                                  │
  Transcript tiếng Việt ────────► PhoBERT (freeze) ───────────► BiLSTM ────────┘
                                  → phobert_dim = 768           → hidden = 256
```

**Đặc điểm:**
- Cần GPU mạnh (ResNet50 + PhoBERT trong memory cùng lúc)
- Cần cài đặt đầy đủ: `torchvision`, `transformers`, `librosa`
- Backbone đóng vai trò trích xuất đặc trưng **từ dữ liệu thô**
- Phù hợp khi: Thu thập dữ liệu tiếng Việt mới bằng EDS → train End-to-End
- File config: `training/config.py` (VideoModuleConfig, AudioModuleConfig, TextModuleConfig)

### 10.2. Kiến Trúc B — Xử Lý Đặc Trưng Đã Trích Xuất (Phase 1 CMU-MOSEI)

Cho Phase 1 Pre-training, đầu vào đã là **vector đặc trưng cấp cao** từ file `.pkl`:

```
  Input đã trích xuất:          Không cần Backbone!          Temporal           Fusion
  ─────────────────────          ────────────────────          ─────────          ──────
  text (50, 768) float32 ──────────────────────────────────► BiLSTM ────────┐
  (BERT embeddings)                                           → hidden = 128  │
                                                                               │
  audio (50, 74) float64 ──────────────────────────────────► BiLSTM ────────┼──► Concat → FC → 1 (regression)
  (COVAREP features)                                          → hidden = 64   │     (256+128+128=512)
                                                                               │
  vision (50, 35) float64 ─────────────────────────────────► BiLSTM ────────┘
  (FACET Action Units)                                        → hidden = 64
```

**Đặc điểm:**
- **Không cần ResNet, CNN, PhoBERT** → chạy được trên GPU yếu hoặc CPU
- Chỉ cần `torch` và `numpy`
- LSTM **trực tiếp nhận vector đặc trưng** → mô hình hóa temporal dynamics
- Đầu ra: **Regression** (sentiment score [-3, +3]) thay vì classification 7 lớp
- Phù hợp khi: Dùng file `.pkl` có sẵn (CMU-MOSEI)

### 10.3. Bảng So Sánh Trực Quan

| Tiêu chí | Kiến Trúc A (Dữ liệu thô) | Kiến Trúc B (Đặc trưng sẵn) |
|:---|:---|:---|
| **Đầu vào** | Ảnh .jpg, .wav, text string | Numpy arrays từ .pkl |
| **Backbone** | ResNet50 + CNN1D + PhoBERT | ❌ Không cần |
| **Temporal** | LSTM sau backbone | LSTM trực tiếp |
| **Fusion input dim** | 512 × 3 = 1,536 | 256 + 128 + 128 = 512 |
| **Đầu ra** | 7 lớp cảm xúc (classification) | 1 số thực [-3, +3] (regression) |
| **Loss** | CrossEntropyLoss | MSELoss |
| **GPU cần** | ≥ 6 GB VRAM | ≥ 2 GB VRAM (hoặc CPU) |
| **Dependencies** | torchvision, transformers, librosa | chỉ torch, numpy |
| **Dùng cho** | Phase 2 (dữ liệu Việt mới) | Phase 1 (CMU-MOSEI .pkl) |
| **Config file** | `training/config.py` | `training/config_phase1.py` (MỚI) |

### 10.4. Mối Quan Hệ Giữa 2 Kiến Trúc

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  Kiến Trúc B (Phase 1)          Kiến Trúc A (Phase 2)          │
  │  ─────────────────────           ─────────────────────           │
  │  Pre-extracted Features          Raw Data End-to-End             │
  │  .pkl → LSTM → Fusion           Video/Audio/Text → Full Model   │
  │         │                                    │                   │
  │         │  checkpoint.pt                     │                   │
  │         └────────────────────────►───────────┘                   │
  │              Transfer Fusion                                     │
  │              weights only                                        │
  │                                                                  │
  │  Cách chuyển đổi:                                               │
  │  1. Train Kiến Trúc B → lưu fusion_network weights              │
  │  2. Khởi tạo Kiến Trúc A → load fusion_network weights          │
  │  3. Freeze LSTM encoders → chỉ train fusion + backbones         │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
```

**Giải thích:** Phần Fusion Network (các lớp Linear + BatchNorm + Dropout cuối cùng) học được **cách kết hợp thông tin đa phương thức**. Kiến thức này có thể transfer từ Kiến Trúc B sang Kiến Trúc A, miễn là kích thước input của Fusion được thiết kế tương thích.

---

## 11. Cấu Hình Chi Tiết & Cấu Trúc Code Phase 1

### 11.1. Cấu Hình Phase 1 (config_phase1.py)

Dưới đây là thiết kế cấu hình riêng cho Phase 1, hoạt động **song song** với `config.py` hiện tại (không xung đột):

```python
# training/config_phase1.py

@dataclass
class MOSEIDataConfig:
    """Cấu hình dữ liệu CMU-MOSEI Phase 1."""
    
    # Đường dẫn file .pkl
    aligned_data_path: str = "data/MSA-Dataset/aligned_50.pkl"
    unaligned_data_path: str = "data/MSA-Dataset/unaligned_50.pkl"
    use_aligned: bool = True        # True = aligned, False = unaligned
    
    # Kích thước đặc trưng (cố định, từ chẩn đoán thực tế)
    text_dim: int = 768             # BERT embedding dimension
    audio_dim: int = 74             # COVAREP feature dimension
    vision_dim: int = 35            # FACET Action Units dimension
    seq_len: int = 50               # Sequence length (word-aligned)
    
    # Tiền xử lý
    text_nan_value: float = 0.0     # Thay NaN bằng giá trị này
    audio_inf_value: float = 0.0    # Thay -inf bằng giá trị này
    normalize: bool = True          # Chuẩn hóa đặc trưng về [0,1]


@dataclass
class Phase1ModelConfig:
    """Cấu hình mô hình LSTM Fusion cho Phase 1."""
    
    # Text LSTM Encoder
    text_input_dim: int = 768
    text_hidden_dim: int = 128
    text_num_layers: int = 1
    text_bidirectional: bool = True
    text_dropout: float = 0.1
    # → output dim = 128 × 2 = 256 (bidirectional)
    
    # Audio LSTM Encoder
    audio_input_dim: int = 74
    audio_hidden_dim: int = 64
    audio_num_layers: int = 1
    audio_bidirectional: bool = True
    audio_dropout: float = 0.1
    # → output dim = 64 × 2 = 128
    
    # Vision LSTM Encoder
    vision_input_dim: int = 35
    vision_hidden_dim: int = 64
    vision_num_layers: int = 1
    vision_bidirectional: bool = True
    vision_dropout: float = 0.1
    # → output dim = 64 × 2 = 128
    
    # Fusion Network
    fusion_input_dim: int = 512     # 256 + 128 + 128
    fusion_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    fusion_dropout: float = 0.3
    output_dim: int = 1             # Regression: 1 số thực


@dataclass
class Phase1TrainingConfig:
    """Cấu hình huấn luyện Phase 1."""
    
    batch_size: int = 32
    learning_rate: float = 1e-3     # Cao hơn Phase 2 vì train từ đầu
    weight_decay: float = 1e-4
    num_epochs: int = 50
    
    optimizer: str = "adam"
    scheduler: str = "plateau"      # ReduceLROnPlateau
    scheduler_patience: int = 3     # Giảm LR sau 3 epoch không cải thiện
    scheduler_factor: float = 0.5   # Giảm LR còn 50%
    
    early_stopping_patience: int = 8
    gradient_clip_norm: float = 1.0
    
    seed: int = 42
    log_every_n_steps: int = 50
    eval_every_n_epochs: int = 1
    save_best_only: bool = True
    
    checkpoint_dir: str = "checkpoints/phase1"
    log_dir: str = "logs/phase1"
```

### 11.2. Cấu Trúc Thư Mục Code Phase 1

```
  training/
  ├── __init__.py                          # ← Đã có
  ├── config.py                            # ← Đã có (cho Kiến Trúc A / Phase 2)
  │
  ├── config_phase1.py                     # [MỚI] Cấu hình riêng Phase 1
  │
  ├── dataset_mosei.py                     # [MỚI] PyTorch Dataset đọc .pkl
  │   ├── class MOSEIDataset(Dataset)
  │   │     load() → đọc pickle, chia train/valid/test
  │   │     __getitem__() → trả về (text, audio, vision, label)
  │   │     _preprocess() → thay -inf, normalize
  │   └── get_dataloaders() → trả về 3 DataLoader
  │
  ├── models/
  │   ├── __init__.py
  │   └── fusion_model.py                  # [MỚI] LSTM Encoders + Fusion
  │       ├── class LSTMEncoder(nn.Module)
  │       │     forward(x) → hidden state cuối
  │       └── class MultimodalFusion(nn.Module)
  │             __init__() → tạo 3 LSTM encoders + fusion FC
  │             forward(text, audio, vision) → sentiment score
  │
  ├── utils/
  │   ├── __init__.py
  │   └── metrics.py                        # [MỚI] MAE, Corr, Acc-2, Acc-7, F1
  │       ├── eval_mosei() → tính tất cả metrics
  │       ├── multiclass_acc() → Acc-N
  │       └── calc_metrics() → dict kết quả
  │
  └── train_phase1.py                      # [MỚI] Script huấn luyện chính
      ├── train_one_epoch() → vòng lặp train
      ├── evaluate() → vòng lặp eval trên valid/test
      ├── main() → setup, train loop, save checkpoint
      └── if __name__ == "__main__": main()
```

### 11.3. Luồng Thực Thi Phase 1

```
  python training/train_phase1.py
       │
       ▼
  1. Load config (Phase1Config)
       │
       ▼
  2. Load aligned_50.pkl → MOSEIDataset
     → Chia train/valid/test DataLoader
     → Thay -inf → 0 trong audio features
       │
       ▼
  3. Khởi tạo MultimodalFusion model
     → TextLSTM(768 → 256)
     → AudioLSTM(74 → 128)
     → VisionLSTM(35 → 128)
     → FusionFC(512 → 256 → 128 → 1)
       │
       ▼
  4. Training loop (50 epochs, early stopping):
     FOR mỗi epoch:
       │
       ├─► Train: model.train()
       │   FOR mỗi batch (32 mẫu):
       │     text, audio, vision, label = batch
       │     output = model(text, audio, vision)     # → ŷ ∈ [-3, +3]
       │     loss = MSELoss(output, label)
       │     loss.backward()
       │     clip_grad_norm_(max=1.0)
       │     optimizer.step()
       │
       ├─► Evaluate: model.eval()
       │   Tính trên valid set:
       │     MAE, Corr, Acc-2, Acc-7, F1
       │
       ├─► Scheduler: ReduceLROnPlateau(val_loss)
       │
       └─► Early Stopping: nếu val_loss không giảm 8 epoch liên tiếp → DỪNG
       │
       ▼
  5. Lưu best_model.pt (theo val_loss thấp nhất)
       │
       ▼
  6. Đánh giá cuối cùng trên test set:
     In kết quả: MAE, Corr, Acc-2, Acc-5, Acc-7, F1
     So sánh với baseline MMSA:
       │ Metric   │ MMSA Baseline │ Mục tiêu Phase 1 │
       │ MAE      │ ~0.58         │ < 0.65            │
       │ Corr     │ ~0.74         │ > 0.68            │
       │ Acc-2    │ ~79%          │ > 75%             │
```

### 11.4. Chi Tiết Hàm Đánh Giá (Metrics)

Metrics đánh giá theo chuẩn MMSA, được tham khảo từ [metricsTop.py](file:///d:/Hai/study/DeepLerning/BCDA/data/MSA-Dataset/Git/MMSA/src/MMSA/utils/metricsTop.py):

```python
# Cách chuyển đổi regression output → accuracy metrics:

def binary_accuracy(preds, labels):
    """Acc-2: Positive (≥0) vs Negative (<0)."""
    # Hai biến thể:
    # Has0 = non-negative: preds ≥ 0 → positive
    pred_binary = (preds >= 0).long()
    true_binary = (labels >= 0).long()
    return (pred_binary == true_binary).float().mean()

def multiclass_accuracy(preds, labels, num_classes=7):
    """Acc-7: Làm tròn về [-3,-2,-1,0,1,2,3]."""
    preds_rounded = torch.clamp(torch.round(preds), min=-3, max=3).long() + 3
    labels_rounded = torch.clamp(torch.round(labels), min=-3, max=3).long() + 3
    return (preds_rounded == labels_rounded).float().mean()

def pearson_correlation(preds, labels):
    """Corr: Hệ số tương quan Pearson."""
    preds_centered = preds - preds.mean()
    labels_centered = labels - labels.mean()
    numerator = (preds_centered * labels_centered).sum()
    denominator = torch.sqrt((preds_centered ** 2).sum() * (labels_centered ** 2).sum())
    return numerator / (denominator + 1e-8)
```

---

## 12. Kế Hoạch Kiểm Thử & Xác Nhận

### 12.1. Kiểm Thử Trước Khi Train

| Bước kiểm thử | Mục đích | Cách thực hiện |
|:---|:---|:---|
| Load `.pkl` | Verify file không lỗi | `pickle.load()` → check keys, shapes, dtypes |
| DataLoader | Verify batch generation | Lấy 1 batch → in shapes, kiểm tra giá trị |
| Model forward | Verify kiến trúc đúng | Input dummy tensors → check output shape = (batch, 1) |
| Loss backward | Verify gradient flow | 1 step train → check loss giảm, gradients ≠ 0 |
| Overfit 1 batch | Verify model học được | Train 100 epochs trên 1 batch → loss → ~0 |

### 12.2. Kiểm Thử Sau Khi Train

| Metric | Baseline MMSA (tham khảo) | Mục tiêu Phase 1 | Đạt? |
|:---|:---|:---|:---|
| MAE ↓ | ~0.58 | < 0.65 | Chạy xong mới biết |
| Corr ↑ | ~0.74 | > 0.68 | |
| Acc-2 ↑ | ~79% | > 75% | |
| Acc-5 ↑ | ~48% | > 42% | |
| Acc-7 ↑ | ~43% | > 38% | |
| F1 ↑ | ~79% | > 75% | |

**Lưu ý:** Baseline MMSA dùng kiến trúc phức tạp hơn (Self-MM, TFN, LMF). Mô hình Phase 1 đơn giản hơn (LSTM + Concat Fusion) nên kỳ vọng kết quả thấp hơn ~5-10%. Nếu kết quả quá thấp, có thể nâng cấp lên TFN/LMF trong cùng framework.
