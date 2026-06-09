# 🚀 HƯỚNG DẪN HUẤN LUYỆN TRÊN GOOGLE COLAB & ĐỒNG BỘ GCS - W&B

Tài liệu này ghi lại chi tiết các bước thiết lập, đồng bộ dữ liệu lên Google Cloud Storage (GCS), cấu hình Weights & Biases (W&B), chạy huấn luyện trên Google Colab và kéo kết quả về máy local.

---

## 📋 MỤC LỤC
1. [Chuẩn Bị local & Đẩy Code Lên GitHub](#1-chuẩn-bị-local--đẩy-code-lên-github)
2. [Upload Dataset Lên GCS Bucket](#2-upload-dataset-lên-gcs-bucket)
3. [Liên Kết Tài Khoản Weights & Biases](#3-liên-kết-tài-khoản-weights--biases)
4. [Các Bước Huấn Luyện Trên Google Colab](#4-các-bước-huấn-luyện-trên-google-colab)
5. [Đồng Bộ Kết Quả Về Máy Local](#5-đồng-bộ-kết-quả-về-máy-local)

---

## 1. Chuẩn Bị Local & Đẩy Code Lên GitHub

Do Google Colab sẽ clone code trực tiếp từ Repository Git của bạn để đảm bảo môi trường đồng bộ, tất cả các cập nhật cấu hình phải được đưa lên GitHub trước.

**Các bước thực hiện:**
1. Mở terminal tại thư mục gốc của dự án `BCDA/`.
2. Thực hiện commit và push:
   ```bash
   # Thêm tất cả các file thay đổi (config, trainer, notebook, sync tool)
   git add .

   # Commit với thông điệp rõ ràng
   git commit -m "feat: complete Phase 1 Colab integration, GCS sync tool, and EDS tool refinements"

   # Đẩy lên GitHub (nhánh main)
   git push origin main
   ```

---

## 2. Upload Dataset Lên GCS Bucket

Để Colab không bị nghẽn mạng hoặc giới hạn băng thông khi đọc từ Google Drive, chúng ta sử dụng GCS Bucket.

**Các bước thực hiện:**
1. Kiểm tra danh sách bucket hiện có:
   ```bash
   gcloud storage buckets list
   ```
2. Nếu chưa có bucket phù hợp, tạo một bucket mới duy nhất (ví dụ đặt tên là `mer-data-bucket-kandesfx` tại khu vực Singapore để có tốc độ tốt nhất):
   ```bash
   gcloud storage buckets create gs://mer-data-bucket-kandesfx --location=asia-southeast1
   ```
3. Upload tệp dữ liệu đã căn chỉnh `aligned_50.pkl` (~4.4 GB) lên bucket:
   ```bash
   gcloud storage cp d:\Hai\study\DeepLerning\BCDA\data\MSA-Dataset\aligned_50.pkl gs://mer-data-bucket-kandesfx/data/MSA-Dataset/aligned_50.pkl
   ```

---

## 3. Liên Kết Tài Khoản Weights & Biases

Weights & Biases (W&B) là công cụ giám sát giúp theo dõi trực quan biểu đồ huấn luyện (Loss, MAE, Correlation...) theo thời gian thực.

**Các bước thực hiện:**
1. Đăng nhập vào tài khoản của bạn tại: [wandb.ai](https://wandb.ai/)
2. Truy cập trang lấy mã API Key: [wandb.ai/authorize](https://wandb.ai/authorize)
3. Nhấn **Copy** để lưu khóa này vào clipboard (khóa này sẽ dùng để đăng nhập trên Colab).

---

## 4. Các Bước Huấn Luyện Trên Google Colab

1. Truy cập Google Colab, mở notebook [02_baseline_early_fusion.ipynb](https://github.com/Kandesfx/Training-Multimodal-Emotion-Analysis/blob/main/notebooks/02_baseline_early_fusion.ipynb) trực tiếp từ repository của bạn.
2. Kiểm tra/Thay đổi các thông số cấu hình ở **Cell số 2**:
   * `USE_GCS = True`
   * `GCS_BUCKET = 'mer-data-bucket-kandesfx'`
   * `WANDB_ENABLE = True`
   * `WANDB_PROJECT = 'bcda-phase1'`
3. **Chạy Cell số 2:**
   * Một cửa sổ pop-up của Google sẽ hiện lên yêu cầu cấp quyền truy cập Cloud SDK. Hãy nhấn **Allow** để cho phép Colab kết nối GCS của bạn.
   * Colab sẽ tự động tải file `aligned_50.pkl` từ GCS về ổ đĩa SSD cục bộ cực kỳ nhanh (~1-2 phút).
4. **Chạy Cell số 4 (Cấu hình model & Đăng nhập W&B):**
   * Khi gọi đến hàm `wandb.login()`, Colab sẽ xuất hiện ô nhập liệu yêu cầu API Key.
   * Hãy **dán (paste)** khóa API Key đã copy ở phần trước vào ô đó và nhấn **Enter**.
5. **Chạy Cell huấn luyện (`trainer.fit(...)`):**
   * Ngay khi quá trình train bắt đầu, Colab sẽ in ra một đường dẫn có dạng:
     `View run at https://wandb.ai/<username>/bcda-phase1/runs/<run_id>`
   * Nhấn vào đường dẫn này để theo dõi biểu đồ Loss đang cập nhật real-time.

---

## 5. Đồng Bộ Kết Quả Về Máy Local

Trong quá trình Colab huấn luyện hoặc sau khi hoàn thành, các tệp checkpoints (`best_model.pt`, `last_model.pt`) và file log (`history.csv`) sẽ được tự động lưu ngược trở lại GCS bucket của bạn.

Để tải chúng về máy local, bạn chỉ cần mở terminal dưới máy local và chạy script đồng bộ tự động:

```bash
# Đồng bộ checkpoints, logs và outputs từ GCS về thư mục local của dự án
python tools/sync_gcs.py --bucket mer-data-bucket-kandesfx --direction down

# (Tùy chọn) Chạy thử không tải thật để kiểm tra tệp tin thay đổi
python tools/sync_gcs.py --bucket mer-data-bucket-kandesfx --direction down --dry-run
```

---

## 6. Đánh Giá Tất Cả Các Mô Hình Đã Train

Sau khi checkpoints đã được đồng bộ về local/GCS, chạy notebook `08_evaluate_all_models.ipynb` để đánh giá tất cả models trên train/valid/test splits.

### 6.1. Checkpoints Hiện Có

| File | Model | Task | Loss | Trạng thái |
|------|-------|------|------|-------------|
| `best_model_mult.pt` | MulT (aligned) | Sentiment | MSE+L1 | ✅ Train tốt |
| `best_model_mult_unaligned.pt` | MulT (unaligned) | Sentiment | MSE+L1 | ✅ Train tốt |
| `best_model_improved_lstm.pt` | Improved LSTM | Sentiment | MSE | ✅ Train tốt |
| `best_model_mult_emotion.pt` | MulT (P0) | Emotion | BCE | ✅ Train tốt |
| `best_model_mult_emotion_p1_focal.pt` | MulT (P1) | Emotion | Focal Loss | ❌ **DIVERGED** — không dùng được |

### 6.2. Kết Quả Emotion P0 (BCE Loss)

**Checkpoint:** `best_model_mult_emotion.pt`

| Split | Mean F1 | Mean Acc | Happy F1 | Sad F1 | Angry F1 | Surprise F1 | Disgust F1 | Fear F1 |
|-------|---------|----------|----------|--------|---------|------------|------------|---------|
| Valid (raw) | 0.2118 | — | — | — | — | — | — | — |
| Test (raw) | **0.2307** | — | **0.4709** | ~0.27 | ~0.19 | ~0.10 | ~0.17 | 0.035 |
| Test (tuned thresh) | **0.2621** | 0.4739 | **0.5984** | 0.2395 | 0.3165 | 0.0558 | 0.3048 | 0.0576 |

> Threshold tuning (P0.1) cải thiện Mean F1 test thêm **+13.6%** mà không cần train lại.

### 6.3. Sentiment — Kết Quả Từ Training Log

| Metric | MMSA MulT (SOTA) | Improved LSTM | Target |
|--------|------------------|-------------|--------|
| MAE | 0.5593 | 0.5859 | ≤ 0.5700 |
| Corr | 0.7331 | 0.7229 | ≥ 0.7300 |

> MulT aligned và unaligned cần chạy evaluate notebook để lấy kết quả thực tế.

### 6.4. Cách Chạy Evaluate

1. Mở notebook `08_evaluate_all_models.ipynb` trên Colab
2. Mount Drive và clone repo (các cell đầu tiên)
3. Chạy cell **"EVALUATE ALL PHASE 1 CHECKPOINTS"**
4. Kết quả: bảng so sánh đầy đủ + file `all_evaluations.json` lưu về Drive

### 6.5. Bug Focal Loss (Đã Fix)

Checkpoint `best_model_mult_emotion_p1_focal.pt` **bị diverged** do 2 bug trong `FocalLoss`:

1. `pos_weight` bị nhân nhầm vào `alpha_t` → loss inflated 30-50x
2. Reduction bất thường (`sum / mean_positives`) → loss tăng phi tuyến

**Fix đã commit:** `1a15119` — FocalLoss dùng `alpha` chuẩn Lin et al. 2017, reduction `.mean()`.

> **Cần retrain P1 emotion** sau fix này.
