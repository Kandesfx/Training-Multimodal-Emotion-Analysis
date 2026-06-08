# CMU-MOSEI 6-Emotion Integration & Multi-Label Training Report

Tài liệu này ghi lại chi tiết vấn đề, giải pháp kỹ thuật, và lịch sử hoạt động tích hợp **6 nhãn cảm xúc** (happy, sad, angry, surprise, disgust, fear) cho mô hình **MulT (Multimodal Transformer)** trên bộ dữ liệu CMU-MOSEI.

---

## 1. Vấn đề & Thách thức (Problems & Challenges)

### Vấn đề gốc
Bộ dữ liệu CMU-MOSEI gốc chứa hai loại nhãn:
1. **Sentiment labels:** Điểm liên tục $[-3, +3]$ (Đã được tích hợp sẵn trong file `.pkl` của MMSA).
2. **Emotion labels:** Cường độ của 6 nhóm cảm xúc $[0, 3]$ (Chưa được tích hợp trong file `.pkl` tiền xử lý).

Để chuyển từ bài toán dự báo Sentiment sang phân loại đa cảm xúc (Multi-label Emotion Classification), chúng ta gặp phải các thách thức lớn sau:

### Thách thức 1: Server tải dữ liệu của CMU bị sập (Downtime)
* **Chi tiết:** Khi sử dụng thư viện `mmsdk` để tải file computational sequence chứa nhãn cảm xúc gốc (`CMU_MOSEI_Labels.csd`), script liên tục gặp lỗi timeout (`ConnectTimeoutError`) do server chính thức của CMU (`immortal.multicomp.cs.cmu.edu`) ngắt kết nối hoặc không phản hồi.

### Thách thức 2: Sự không đồng bộ về ID (ID Misalignment)
* **Chi tiết:** Định dạng ID mẫu trong file `.pkl` của MMSA (`{youtube_id}$_${segment_number}`) và định dạng trong SDK gốc có sự sai lệch nhỏ về cách đếm segment. Điều này dẫn đến việc không thể khớp nhãn 100%. Coverage thực tế đạt khoảng **76.5% - 77.5%**.

### Thách thức 3: Nhiễu nhãn Trung tính giả (False Neutrals Noise)
* **Chi tiết:** Với ~23% số mẫu không khớp được nhãn, nếu huấn luyện trực tiếp, các mẫu này sẽ nhận nhãn mặc định là vector không `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` (tương đương với Neutral). Điều này khiến mô hình bị phạt sai khi dự đoán đúng cảm xúc của các mẫu này, gây loãng hàm Loss và suy giảm chất lượng dự báo F1-score.

---

## 2. Giải pháp Kỹ thuật (Technical Solutions)

Chúng ta đã giải quyết triệt để 3 thách thức trên bằng các giải pháp sau:

### Giải pháp 1: Tải dữ liệu từ Hugging Face Mirror & Hỗ trợ Offline Caching
* Trực tiếp tải file `CMU_MOSEI_Labels.csd` từ mirror uy tín của cộng đồng: [reeha-parkar/cmu-mosei-comp-seq](https://huggingface.co/datasets/reeha-parkar/cmu-mosei-comp-seq).
* Nâng cấp [download_emotion_labels.py](file:///d:/Hai/study/DeepLerning/BCDA/scripts/download_emotion_labels.py) để ưu tiên đọc file cache cục bộ nếu đã tải xuống trước đó, và bổ sung bắt lỗi ngoại lệ `Exception` (thay vì chỉ catch `RuntimeError`) để chuyển hướng fallback chính xác.

### Giải pháp 2: Trộn dữ liệu kèm Mặt nạ Khớp nhãn (Matched Mask)
* Cải tiến [merge_emotions_to_pkl.py](file:///d:/Hai/study/DeepLerning/BCDA/scripts/merge_emotions_to_pkl.py) để không chỉ lưu `emotion_labels` mà còn lưu một vector boolean **`emotion_matched_mask`** nhằm đánh dấu chính xác các mẫu nào được khớp nhãn thành công.

### Giải pháp 3: Tự động Lọc nhiễu ở mức Dataset API (Noise Filtering)
* Cập nhật lớp `MOSEIAlignedDataset` và `MOSEIUnalignedDataset` trong [dataset_mosei.py](file:///d:/Hai/study/DeepLerning/BCDA/training/dataset_mosei.py).
* Khi đặt chế độ **`task_type = "emotion"`**, Dataset sẽ tự động đọc `emotion_matched_mask` và lọc bỏ hoàn toàn các mẫu không khớp khỏi pipeline huấn luyện/đánh giá. 
* **Kết quả:** Tập huấn luyện giảm từ `16,326` xuống còn `12,484` mẫu sạch 100%, loại bỏ hoàn toàn các nhãn giả trung tính.

> [!IMPORTANT]
> Cơ chế lọc tự động này giúp bảo toàn tính chính xác của các chỉ số đánh giá (F1, Accuracy), ngăn chặn mô hình bị thiên lệch (bias) về hướng dự đoán Neutral do nhiễu dữ liệu.

---

## 3. Lịch sử Hoạt động & Các Tệp đã Cập nhật (Activity Log)

Dưới đây là chi tiết các file đã sửa đổi và tạo mới được đẩy lên GitHub (Commit `e18fd3f`):

| Thành phần / Tệp | Loại | Nội dung thay đổi |
|:---|:---:|:---|
| [config_phase1.py](file:///d:/Hai/study/DeepLerning/BCDA/training/config_phase1.py) | **MODIFY** | Thêm trường `task_type = "sentiment" \| "emotion"` vào cấu hình chung. |
| [mult.py](file:///d:/Hai/study/DeepLerning/BCDA/training/models/mult.py) | **MODIFY** | Tối ưu hóa Regressor head của MulT để hỗ trợ linh hoạt `output_dim` (1 cho sentiment, 6 cho emotion) và chỉ squeeze chiều cuối khi `output_dim == 1`. |
| [dataset_mosei.py](file:///d:/Hai/study/DeepLerning/BCDA/training/dataset_mosei.py) | **MODIFY** | Thêm nhãn `emotion_labels` vào hàm trả về dạng tensor `(6,)` và triển khai bộ lọc tự động loại bỏ mẫu nhiễu dựa trên `emotion_matched_mask`. |
| [trainer.py](file:///d:/Hai/study/DeepLerning/BCDA/training/trainer.py) | **MODIFY** | Hỗ trợ cấu hình `BCEWithLogitsLoss` cho tác vụ phân loại đa nhãn và dispatch luồng đánh giá sang bộ đo emotion chuyên biệt. |
| [evaluator_emotion.py](file:///d:/Hai/study/DeepLerning/BCDA/training/evaluator_emotion.py) | **NEW** | Viết bộ đánh giá hiệu năng cảm xúc: tính toán F1-score, Accuracy riêng lẻ từng nhãn (happy, sad, angry...) và các giá trị trung bình (mean F1, mean Acc, mean MAE). |
| [download_emotion_labels.py](file:///d:/Hai/study/DeepLerning/BCDA/scripts/download_emotion_labels.py) | **MODIFY** | Thêm cơ chế ưu tiên cache offline và sửa lỗi ngoại lệ tải mạng. |
| [merge_emotions_to_pkl.py](file:///d:/Hai/study/DeepLerning/BCDA/scripts/merge_emotions_to_pkl.py) | **MODIFY** | Cập nhật lưu trữ thêm mặt nạ boolean `emotion_matched_mask`. |
| [05_mult_emotion_training.ipynb](file:///d:/Hai/study/DeepLerning/BCDA/notebooks/05_mult_emotion_training.ipynb) | **NEW** | Notebook hoàn chỉnh được cấu hình tối ưu để huấn luyện MulT đa cảm xúc trên Google Colab. |

---

## 4. Thống kê Dữ liệu Thực tế sau khi Gộp

Sau khi chạy cục bộ scripts và gộp thành công vào 3 file dữ liệu (`aligned_50.pkl`, `unaligned_50.pkl`, `aligned_50_vi.pkl`), thống kê phân bố cảm xúc trong tập huấn luyện sạch (**12,484 mẫu**) như sau:

* **Happy (Vui vẻ):** **4,248 mẫu** (34.0% tập sạch), cường độ trung bình `0.92/3.0`
* **Sad (Buồn bã):** **1,765 mẫu** (14.1% tập sạch), cường độ trung bình `0.63/3.0`
* **Angry (Tức giận):** **1,804 mẫu** (14.5% tập sạch), cường độ trung bình `0.77/3.0`
* **Disgust (Ghê tởm):** **1,483 mẫu** (11.9% tập sạch), cường độ trung bình `0.71/3.0`
* **Surprise (Ngạc nhiên):** **410 mẫu** (3.3% tập sạch), cường độ trung bình `0.49/3.0`
* **Fear (Sợ hãi):** **280 mẫu** (2.2% tập sạch), cường độ trung bình `0.49/3.0`

---

## 5. Hướng dẫn Tái tạo & Huấn luyện trên Google Colab

> [!TIP]
> Hãy thực hiện các bước sau khi chạy trên môi trường Google Colab để đảm bảo đồng bộ:

1. **Đồng bộ code mới:**
   ```bash
   git pull origin main
   ```
2. **Cập nhật tệp Pickle lên Cloud Storage (GCS):**
   Vì các tệp `.pkl` có dung lượng rất lớn (~4.6GB đến ~13.6GB) không thể lưu trên Git, hãy tải các file `.pkl` đã xử lý cục bộ lên Google Cloud Storage:
   ```bash
   gcloud storage cp d:\Hai\study\DeepLerning\BCDA\data\MSA-Dataset\aligned_50.pkl gs://mer-data-bucket-kandesfx/data/MSA-Dataset/aligned_50.pkl
   gcloud storage cp d:\Hai\study\DeepLerning\BCDA\data\MSA-Dataset\aligned_50_vi.pkl gs://mer-data-bucket-kandesfx/data/MSA-Dataset/aligned_50_vi.pkl
   gcloud storage cp d:\Hai\study\DeepLerning\BCDA\data\MSA-Dataset\unaligned_50.pkl gs://mer-data-bucket-kandesfx/data/MSA-Dataset/unaligned_50.pkl
   ```
3. **Chạy notebook huấn luyện:**
   Mở và chạy file [05_mult_emotion_training.ipynb](file:///d:/Hai/study/DeepLerning/BCDA/notebooks/05_mult_emotion_training.ipynb) để bắt đầu quá trình train mô hình đa cảm xúc.
