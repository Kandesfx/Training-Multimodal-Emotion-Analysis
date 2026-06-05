# 📊 Báo Cáo Nghiên Cứu & Chẩn Đoán Chi Tiết Tập Dữ Liệu CMU-MOSEI (MMSA)

Tài liệu này cung cấp kết quả chẩn đoán thực tế trên các tệp dữ liệu đã tải xuống và phân tích chi tiết quy trình xử lý, đồng bộ hóa (align) để tạo ra tập dữ liệu đa phương thức chuẩn bị cho huấn luyện mô hình.

---

## 1. Kết Quả Chẩn Đoán Thực Tế Trên Đĩa Cứng

Dưới đây là thông số kỹ thuật thực tế được trích xuất trực tiếp từ các tệp dữ liệu nằm trong thư mục `d:\Hai\study\DeepLerning\BCDA\data\MSA-Dataset\`.

### 1.1. Cấu Trúc File Đóng Gói Sẵn (`.pkl`)

Cả hai tệp `aligned_50.pkl` và `unaligned_50.pkl` được đóng gói bằng thư viện `pickle` của Python dưới dạng một Dictionary lớn chứa 3 phân đoạn dữ liệu chính: `train`, `valid`, và `test`.

#### A. File Căn Chỉnh Theo Từ: `aligned_50.pkl`
* **Dung lượng:** ~1.2 GB
* **Số lượng mẫu mỗi tập (Train / Valid / Test):**
  * `train`: **16,326** mẫu
  * `valid`: **1,871** mẫu
  * `test`: **4,659** mẫu
  * **Tổng cộng:** **22,856** mẫu hội thoại (utterances).
* **Cấu trúc phím (Keys) và định dạng dữ liệu trong mỗi tập:**
  * `'id'`: Kiểu danh sách (`list`) gồm các chuỗi định danh mẫu (Ví dụ: `'-3g5yACwYnA$_$10'`).
  * `'raw_text'`: Mảng numpy (`ndarray`) chứa chuỗi văn bản tiếng Anh gốc thô. Kích thước: `(N,)`.
  * `'text'`: Mảng numpy đặc trưng từ BERT (tầng ẩn cuối). Kích thước: `(N, 50, 768)`, kiểu `float32`.
  * `'text_bert'`: Mảng numpy chứa các tokenized IDs cho mô hình BERT. Kích thước: `(N, 3, 50)`, kiểu `int64`.
  * `'audio'`: Mảng numpy đặc trưng âm thanh COVAREP. Kích thước: `(N, 50, 74)`, kiểu `float64`.
  * `'vision'`: Mảng numpy đặc trưng hình ảnh khuôn mặt FACET. Kích thước: `(N, 50, 35)`, kiểu `float64`.
  * `'regression_labels'`: Mảng numpy nhãn sắc thái liên tục. Kích thước: `(N,)`, kiểu `float64`.
  * `'classification_labels'`: Mảng numpy nhãn phân loại sắc thái 3 lớp. Kích thước: `(N,)`, kiểu `float64`.
  * `'annotations'`: Danh sách (`list`) nhãn từ dạng văn bản (Ví dụ: `'Positive'`, `'Neutral'`, `'Negative'`).

#### B. File Chưa Căn Chỉnh Tần Số: `unaligned_50.pkl`
* **Dung lượng:** ~3.4 GB
* **Số lượng mẫu:** Giống hệt bản aligned (16,326 / 1,871 / 4,659).
* **Điểm khác biệt về kích thước ma trận:**
  * `'text'`: Giữ nguyên kích thước `(N, 50, 768)`.
  * `'audio'`: Được đệm (padding) về kích thước cố định là **`(N, 500, 74)`** thay vì 50.
  * `'vision'`: Được đệm về kích thước cố định là **`(N, 500, 35)`** thay vì 50 (hoặc 375 như tài liệu lý thuyết).
  * `'audio_lengths'`: Danh sách độ dài chuỗi âm thanh thực tế của từng mẫu trước khi đệm (Min: 2, Max: 500, Trung bình: 149.2).
  * `'vision_lengths'`: Danh sách độ dài chuỗi hình ảnh thực tế của từng mẫu trước khi đệm (Min: 1, Max: 500, Trung bình: 94.6).

---

### 1.2. Cấu Trúc File Video Thô: `Raw.zip`
* **Dung lượng:** ~25.4 GB (26,025.56 MB)
* **Tổng số phần tử bên trong:** **26,082** phần tử (bao gồm cả thư mục).
* **Tổng số file video `.mp4`:** Đúng **22,856** file, khớp tỷ lệ 1:1 hoàn hảo với tổng số mẫu trong file pickle.
* **Cấu trúc thư mục lưu trữ:**
  ```text
  Raw/
  └── <YouTube-Video-ID>/
      ├── <Utterance-Index>.mp4
      └── ...
  ```
  * Ví dụ: Một đoạn video cắt nằm tại đường dẫn: `Raw/-3g5yACwYnA/10.mp4`.
* **Cơ chế bản đồ liên kết (ID Mapping):**
  Mỗi mẫu trong file pickle có một khóa `'id'` đại diện, ví dụ: `'-3g5yACwYnA$_$10'`. Khóa này được phân tách bằng ký tự `$_$`. Phần bên trái là **YouTube Video ID** (`-3g5yACwYnA`), và phần bên phải là **Utterance Index** (`10`).
  Từ khóa ID này, ta có thể suy ra trực tiếp đường dẫn đến file video gốc trong `Raw.zip`:
  $$\text{ID: } \texttt{ID\_A\$\_\$Index\_B} \implies \text{Video Path: } \texttt{Raw/ID\_A/Index\_B.mp4}$$

---

## 2. Phân Tích Chi Tiết Định Dạng Nhãn (Labels) & Nhãn 6 Cảm Xúc

Kết quả chẩn đoán nhãn thực tế mang lại những thông tin rất quan trọng để cấu hình đầu ra cho mô hình:

### 2.1. Nhãn Sắc Thái Hồi Quy (`regression_labels`)
* **Định dạng:** Các số thực (`float64`) nằm trong khoảng từ **-3.0** (cực kỳ tiêu cực) đến **+3.0** (cực kỳ tích cực).
* **Giá trị trung bình của tập train:** `0.1463` (cho thấy tập dữ liệu tương đối cân bằng, có xu hướng hơi tích cực nhẹ).
* **Vai trò:** Đây là nhãn mục tiêu chính dùng cho huấn luyện mô hình theo bài toán hồi quy (Regression).

### 2.2. Nhãn Phân Loại Sắc Thái (`classification_labels` & `annotations`)
* Nhãn `'classification_labels'` chứa các giá trị thực `[0.0, 1.0, 2.0]`. Nhãn `'annotations'` tương ứng chứa các chuỗi `['Negative', 'Neutral', 'Positive']`.
* Quy tắc ánh xạ thực tế:
  * **0.0 (`Negative`):** Khi điểm hồi quy $< 0$ (Ví dụ: $-0.666 \implies 0.0$).
  * **1.0 (`Neutral`):** Khi điểm hồi quy $= 0$ (Ví dụ: $0.0 \implies 1.0$).
  * **2.0 (`Positive`):** Khi điểm hồi quy $> 0$ (Ví dụ: $+0.333 \implies 2.0$, $+1.0 \implies 2.0$).
* **Lưu ý quan trọng (MMSA v2.0):** Nhãn `classification_labels` đã bị **đánh dấu lỗi thời (deprecated)** trong khung MMSA v2.0. Lý do là mô hình sẽ được huấn luyện trực tiếp bằng đầu ra Hồi quy (Regression). Toàn bộ các chỉ số phân loại (2-class, 3-class, 5-class, 7-class) sẽ được tính toán gián tiếp bằng cách làm tròn hoặc phân ngưỡng từ đầu ra Hồi quy trong quá trình đánh giá (Evaluation).

### 2.3. Về Nhãn 6 Cảm Xúc Cơ Bản (Happiness, Sadness, Anger, Surprise, Disgust, Fear)
* **Chẩn đoán:** Khung dữ liệu preprocessed của MMSA **chỉ tập trung vào Sentiment (Sắc thái)**. Vì vậy, các file `.pkl` của MMSA tải về **không chứa sẵn** khóa nhãn 6 cảm xúc (`emotions`).
* **Hướng xử lý:**
  1. Nếu mục tiêu cốt lõi là **Phân tích Sắc thái (Sentiment)**: Sử dụng trực tiếp `regression_labels` làm nhãn huấn luyện mục tiêu.
  2. Nếu bắt buộc phải huấn luyện **Nhận diện 6 cảm xúc**: Ta cần phải tải các file nhãn cảm xúc `.csd` gốc từ **CMU-MultimodalSDK** hoặc sử dụng file nhãn cảm xúc đã đồng bộ từ các dự án khác bên ngoài MMSA.

---

## 3. Cấu Trúc Khóa BERT Đặc Biệt (`text_bert`)

Trong file `.pkl`, khóa `'text_bert'` có kích thước **`(N, 3, 50)`** chứa thông tin đầu vào chuẩn hóa của BERT:

```text
Kích thước (N, 3, 50) của mỗi mẫu:
┌────────────────────────────────────────────────────────┐
│ Hàng 0: Token IDs (101 = [CLS], 102 = [SEP], 0 = [PAD])│ -> Chiều dài 50
├────────────────────────────────────────────────────────┤
│ Hàng 1: Attention Mask (1 cho token thực, 0 cho đệm)   │ -> Chiều dài 50
├────────────────────────────────────────────────────────┤
│ Hàng 2: Segment/Token Type IDs (Mặc định toàn bộ là 0) │ -> Chiều dài 50
└────────────────────────────────────────────────────────┘
```

Thông tin này giúp chúng ta có thể lựa chọn:
* Hoặc đưa trực tiếp vector đặc trưng tĩnh `(N, 50, 768)` từ khóa `'text'` vào mô hình mạng nơ-ron tích hợp (Fusion Network).
* Hoặc nạp trực tiếp `text_bert` vào mô hình BERT gốc (đã unfreeze) để huấn luyện tinh chỉnh (Fine-tune End-to-End) cả nhánh ngôn ngữ.

---

## 4. Phương Pháp Tiền Xử Lý & Căn Chỉnh Dữ Liệu (Alignment Pipeline)

Để tạo ra các file dữ liệu sạch từ tập video thô `Raw.zip`, các tác giả của CMU và Thanh Hoa đã thực hiện quy trình kỹ thuật như sau:

```text
   +-------------+       +-------------------+       +-----------------------+
   |  Raw Video  | ----> | Trích xuất Audio  | ----> | Đặc trưng COVAREP     | --+
   |   (.mp4)    |       |     (.wav)        |       | (74 chiều - 100Hz)    |   |
   +-------------+       +-------------------+       +-----------------------+   |
          |                                                                      |
          +------------> | Trích xuất Khung  | ----> | Đặc trưng FACET (AUs) | --+
          |              | hình (Frames)     |       | (35 chiều - 30Hz)     |   |
          |                                                                      v
          +------------> | Trích xuất Lời    | ----> | Mốc thời gian từ đơn  | --+ [Đồng bộ hóa]
                         | (Transcripts)     |       | (Forced Alignment)    |     | (Mean Pooling)
                         +-------------------+       +-----------------------+     v
                                                                             +-----------+
                                                                             | Aligned   |
                                                                             |  Features |
                                                                             +-----------+
```

### 4.1. Chi Tiết Phương Pháp Trích Xuất Đặc Trưng (Feature Extraction)
* **Văn bản (Text):** 
  * Sử dụng mô hình **BERT-base-uncased** tiền huấn luyện.
  * Mỗi từ đơn trong câu thoại được đưa qua BERT để trích xuất vector đặc trưng ẩn ở tầng cuối cùng có kích thước **768 chiều**.
* **Âm thanh (Audio):** 
  * Tách luồng âm thanh từ video sang định dạng sóng `.wav` đơn kênh tần số 16kHz.
  * Sử dụng bộ công cụ **COVAREP** để trích xuất các đặc trưng âm sắc tần số thấp tại tần số lấy mẫu **100Hz** (100 khung hình/giây).
  * Đặc trưng thu được gồm **74 chiều**: Tần số cơ bản F0, các tham số nguồn thanh quản (glottal source parameters), hệ số MFCCs, tỉ lệ hài trên nhiễu (HNR).
* **Hình ảnh (Vision):**
  * Quét qua các khung hình video để phát hiện và căn chỉnh khuôn mặt.
  * Sử dụng phần mềm thương mại **FACET** để trích xuất cường độ xuất hiện của **35 Action Units (AUs)** tương ứng với sự chuyển động co giãn của các vùng cơ mặt cơ bản. Tần số lấy mẫu khoảng **30Hz** (tương ứng tốc độ khung hình video).

### 4.2. Phương Pháp Căn Chỉnh Cấp Độ Từ (Word-level Alignment)
Để giải quyết sự lệch pha tần số lấy mẫu (Văn bản đếm theo từ đơn, Audio 100Hz, Video 30Hz), thuật toán thực hiện đồng bộ hóa thời gian:
1. **Forced Alignment (Căn chỉnh cưỡng bức):** Sử dụng công cụ P2FA hoặc Gentle để phân tích tệp âm thanh `.wav` và văn bản, xác định chính xác thời gian bắt đầu ($t_{start}$) và kết thúc ($t_{end}$) của từng từ đơn được phát âm.
2. **Cắt đoạn theo từ (Segmentation):** Với mỗi từ đơn thứ $i$, trích xuất tất cả các khung hình đặc trưng âm thanh (Audio) và hình ảnh (Video) nằm trong khoảng thời gian $[t_{start}, t_{end}]$.
3. **Trung bình hóa thời gian (Mean Pooling):**
   * Nếu khoảng thời gian chứa 30 khung đặc trưng âm thanh, ta tính trung bình cộng của 30 vector đó để tạo ra **1 vector duy nhất có kích thước 74 chiều** đại diện cho từ đó.
   * Nếu khoảng thời gian chứa 5 khung đặc trưng hình ảnh, ta tính trung bình cộng của 5 vector đó để tạo ra **1 vector duy nhất có kích thước 35 chiều**.
4. **Chuẩn hóa chiều dài (Zero-Padding & Truncation):**
   * Đối với bản **Aligned** (`aligned_50.pkl`): Toàn bộ câu thoại được căn chỉnh đồng đều về độ dài chuỗi tối đa là **50 từ**. Nếu câu nói ngắn hơn 50 từ, thực hiện đệm vector số 0 vào cuối (Zero-Padding). Nếu dài hơn, cắt bớt (Truncation). Kết quả thu được ma trận đặc trưng đồng nhất cho cả 3 phương thức có kích thước chuỗi là 50.

---

## 5. Xây Dựng Bộ Đọc Dữ Liệu PyTorch (Dataset & DataLoader) Chuẩn Xác

Dựa trên kết quả chẩn đoán thực tế về tên khóa và định dạng dữ liệu, dưới đây là lớp `Dataset` PyTorch được thiết kế chuẩn xác để nạp các file `.pkl` này mà không gặp lỗi lệch khóa:

```python
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CMUMOSEIDataset(Dataset):
    def __init__(self, pkl_path, mode='train', need_align=True):
        """
        Bộ đọc dữ liệu chuyên biệt cho CMU-MOSEI từ file MMSA .pkl
        
        Args:
            pkl_path (str): Đường dẫn tuyệt đối đến file .pkl trên đĩa.
            mode (str): Phân đoạn cần nạp ('train', 'valid', hoặc 'test').
            need_align (bool): Đọc từ dữ liệu đã aligned (True) hay unaligned (False).
        """
        assert os.path.exists(pkl_path), f"Không tìm thấy file dữ liệu tại: {pkl_path}"
        self.mode = mode
        self.need_align = need_align
        
        print(f"Đang nạp dữ liệu phân đoạn '{mode}' từ {os.path.basename(pkl_path)}...")
        with open(pkl_path, 'rb') as f:
            full_data = pickle.load(f)
        
        self.data = full_data[mode]
        
        # Đặc trưng các phương thức
        self.text_features = self.data['text']              # Shape: (N, 50, 768)
        self.text_bert_inputs = self.data['text_bert']      # Shape: (N, 3, 50)
        self.audio_features = self.data['audio']            # Shape: (N, 50, 74) hoặc (N, 500, 74)
        self.vision_features = self.data['vision']          # Shape: (N, 50, 35) hoặc (N, 500, 35)
        
        # Nhãn mục tiêu chính
        self.regression_labels = self.data['regression_labels']      # Shape: (N,)
        self.classification_labels = self.data['classification_labels']  # Shape: (N,)
        self.ids = self.data['id']                          # Danh sách chuỗi ID
        
        # Đặc trưng độ dài thực tế (Chỉ có ở bản unaligned)
        if not self.need_align:
            self.audio_lengths = self.data['audio_lengths']
            self.vision_lengths = self.data['vision_lengths']
            
        print(f"Nạp thành công! Số lượng mẫu: {len(self.ids)}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Chuyển đổi đặc trưng sang PyTorch Tensor
        text = torch.tensor(self.text_features[idx], dtype=torch.float32)
        text_bert = torch.tensor(self.text_bert_inputs[idx], dtype=torch.long)
        audio = torch.tensor(self.audio_features[idx], dtype=torch.float32)
        vision = torch.tensor(self.vision_features[idx], dtype=torch.float32)
        
        # Xử lý nhãn
        label_reg = torch.tensor(self.regression_labels[idx], dtype=torch.float32)
        label_cls = torch.tensor(self.classification_labels[idx], dtype=torch.long)
        
        sample = {
            'id': self.ids[idx],
            'text': text,
            'text_bert': text_bert,
            'audio': audio,
            'vision': vision,
            'regression_label': label_reg,
            'classification_label': label_cls
        }
        
        # Bổ sung độ dài thực tế nếu dùng mô hình xử lý chuỗi động (Unaligned LSTM / RNN)
        if not self.need_align:
            sample['audio_len'] = self.audio_lengths[idx]
            sample['vision_len'] = self.vision_lengths[idx]
            
        return sample

# === KHỞI TẠO VÀ CHẠY THỬ DATALOADER ===
if __name__ == '__main__':
    aligned_path = r"d:\Hai\study\DeepLerning\BCDA\data\MSA-Dataset\aligned_50.pkl"
    unaligned_path = r"d:\Hai\study\DeepLerning\BCDA\data\MSA-Dataset\unaligned_50.pkl"
    
    # 1. Chạy thử bản Aligned
    train_dataset_aligned = CMUMOSEIDataset(aligned_path, mode='train', need_align=True)
    dataloader_aligned = DataLoader(train_dataset_aligned, batch_size=4, shuffle=True)
    
    print("\n--- Kiểm tra 1 Batch mẫu của Aligned Dataloader ---")
    for batch in dataloader_aligned:
        print("Mã ID mẫu đầu tiên trong batch:", batch['id'][0])
        print("Đặc trưng Text (BERT) shape:", batch['text'].shape)         # (Batch, 50, 768)
        print("Đặc trưng Text BERT Tokens shape:", batch['text_bert'].shape) # (Batch, 3, 50)
        print("Đặc trưng Audio (COVAREP) shape:", batch['audio'].shape)     # (Batch, 50, 74)
        print("Đặc trưng Video (FACET) shape:", batch['vision'].shape)      # (Batch, 50, 35)
        print("Nhãn Hồi quy (Regression) shape:", batch['regression_label'].shape) # (Batch,)
        print("Nhãn Phân loại (Classification) shape:", batch['classification_label'].shape) # (Batch,)
        break
        
    # 2. Chạy thử bản Unaligned
    print("\n" + "="*60)
    train_dataset_unaligned = CMUMOSEIDataset(unaligned_path, mode='train', need_align=False)
    dataloader_unaligned = DataLoader(train_dataset_unaligned, batch_size=4, shuffle=True)
    
    print("\n--- Kiểm tra 1 Batch mẫu của Unaligned Dataloader ---")
    for batch in dataloader_unaligned:
        print("Đặc trưng Audio (Chưa Căn Chỉnh) shape:", batch['audio'].shape)     # (Batch, 500, 74)
        print("Đặc trưng Video (Chưa Căn Chỉnh) shape:", batch['vision'].shape)    # (Batch, 500, 35)
        print("Độ dài Audio thực tế của mẫu đầu:", batch['audio_len'][0].item())
        print("Độ dài Video thực tế của mẫu đầu:", batch['vision_len'][0].item())
        break
```

---

## 6. Phương Pháp Đánh Giá Kết Quả Từ Đầu Ra Hồi Quy (MMSA Core Metrics)

Để đánh giá chính xác độ chính xác của mô hình khi huấn luyện bằng đầu ra hồi quy liên tục ($\hat{y} \in [-3.0, 3.0]$), chúng ta áp dụng logic tính toán các chỉ số của tác giả trong mã nguồn `metricsTop.py`:

1. **MAE (Mean Absolute Error):** Đo khoảng cách trung bình $L1$ giữa dự đoán và nhãn thực tế:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|$$
2. **Corr (Pearson Correlation):** Đo độ tương quan tuyến tính giữa dự đoán và nhãn thực tế.
3. **Phân loại Nhị phân 2 lớp (Binary Classification - Has0 & Non0):**
   * **Has0 (Gồm nhãn Neutral):** Dự đoán đúng nếu ($\hat{y} \ge 0$ và $y \ge 0$) hoặc ($\hat{y} < 0$ và $y < 0$).
   * **Non0 (Bỏ qua nhãn Neutral):** Chỉ tính toán trên các mẫu có nhãn thực tế $y \neq 0$. Dự đoán đúng nếu ($\hat{y} > 0$ và $y > 0$) hoặc ($\hat{y} < 0$ và $y < 0$).
4. **Phân loại 5 lớp (Multiclass 5-Class Accuracy):** 
   Cắt giá trị dự đoán và nhãn thực tế về khoảng `[-2.0, 2.0]`, sau đó làm tròn về số nguyên gần nhất để đưa về 5 hộp nhãn `[-2, -1, 0, 1, 2]`. Tính tỷ lệ chính xác.
5. **Phân loại 7 lớp (Multiclass 7-Class Accuracy):**
   Cắt giá trị dự đoán và nhãn thực tế về khoảng `[-3.0, 3.0]`, làm tròn về số nguyên gần nhất để đưa về 7 hộp nhãn `[-3, -2, -1, 0, 1, 2, 3]`.

---

## 7. Thiết Lập Bản Địa Hóa Sang Tiếng Việt (Vietnamese Localization Strategy)

Để xây dựng hệ thống nhận diện cảm xúc đa phương thức hoạt động hiệu quả cho **tiếng Việt**, chúng ta thay thế nhánh Văn bản từ tiếng Anh sang tiếng Việt theo chiến lược:

```text
               +-------------------------------------------------+
               |              Tiền Xử Lý Văn Bản                 |
               +-------------------------------------------------+
                                      |
       +------------------------------+------------------------------+
       v                                                             v
[Cách 1: Trích xuất Tĩnh (Static)]                 [Cách 2: Tinh chỉnh Động (Dynamic)]
- Dịch `raw_text` tiếng Anh sang tiếng Việt.       - Sử dụng PhoBERT Tokenizer trực tiếp.
- Đưa qua PhoBERT pre-trained để trích             - Lưu trữ chuỗi `input_ids` và
  xuất vector đặc trưng ẩn 768 chiều.                `attention_mask` tiếng Việt.
- Thay thế hoàn toàn ma trận `'text'` cũ.          - Đưa vào mô hình PhoBERT đóng băng
- Lưu lại thành file `.pkl` tiếng Việt mới.         trọng số và tinh chỉnh 2 lớp cuối.
```

* **Công cụ khuyên dùng:** Mô hình **PhoBERT-base** (`vinai/phobert-base`) với 768 chiều đầu ra, hoàn toàn tương thích và khớp số chiều với kiến trúc Fusion mạng cũ của CMU-MOSEI (vốn thiết kế cho BERT 768 chiều). Điều này cho phép chúng ta tái sử dụng toàn bộ nhánh Audio, Video và cấu trúc Fusion Network mà không cần sửa đổi bất kỳ số chiều đầu vào nào của mô hình!
