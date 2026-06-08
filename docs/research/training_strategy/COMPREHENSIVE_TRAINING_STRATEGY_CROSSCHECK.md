# Báo Cáo Cross-Check: Chiến Lược Training vs Thực Tế Code

**Đề Tài 17: Hệ Thống Phân Tích Cảm Xúc Đa Phương Thức**

**Ngày soạn:** 2026-06-08
**Loại:** Báo cáo phản biện & xác minh code
**Trạng thái:** Hoàn thành

---

## Mục Lục

1. [Tổng quan](#1-tổng-quan)
2. [Cross-check từng điểm phản biện](#2-cross-check-từng-điểm-phản-biện)
3. [Issues mới phát hiện ngoài review](#3-issues-mới-phát-hiện-ngoài-review)
4. [Danh sách fix theo ưu tiên](#4-danh-sách-fix-theo-ưu-tiên)
5. [Kết luận](#5-kết-luận)

---

## 1. Tổng quan

Tài liệu này cross-check chi tiết giữa:
- **COMPREHENSIVE_TRAINING_STRATEGY_REVIEW.md** — nhận xét phản biện từ chuyên gia
- **Code thực tế** — `trainer.py`, `evaluator_emotion.py`, `config_phase1.py`, notebook 03 & 05

**Nguồn files kiểm tra:**
| File | Mục đích |
|:---|:---|
| `training/config_phase1.py` | Config P0+P1 |
| `training/trainer.py` | Training loop, evaluation, logging |
| `training/evaluator_emotion.py` | Emotion metrics computation |
| `training/mult.py` | MulT model implementation |
| `notebooks/03_mult_training.ipynb` | Sentiment training notebook |
| `notebooks/05_mult_emotion_training.ipynb` | Emotion training notebook |
| GCS bucket `mer-data-bucket-kandesfx` | Dataset files |

---

## 2. Cross-check từng điểm phản biện

### 2.1. Phản biện 1: Bản chất bài toán Emotion — Classification vs Regression

| Khía cạnh | Review đề xuất | Code thực tế | Xác minh |
|:---|:---|:---|:---|
| Task definition | **Multi-label Classification** — tập trung Mean F1 | `compute_emotion_metrics()` tính F1 per-emotion + mean F1 | ✅ **Đúng** |
| Loss function | BCEWithLogitsLoss hoặc Focal Loss | `trainer.py:96-97` — BCEWithLogitsLoss | ✅ **Đúng** |
| MAE supplementary | MAE chỉ dùng tham khảo, không dùng cho model selection | Metric chính là `mean_f1`, MAE là `mean_mae` trong output | ✅ **Đúng** |
| Metric for best (sentiment) | `mae` | `config_phase1.py:136` — `metric_for_best: "mae"` | ✅ **Đúng** |
| Metric for best (emotion) | `mean_f1` | Config không có riêng, dùng chung `mae` | ⚠️ **CẦN FIX** — emotion nên dùng `mean_f1` |

**Chi tiết issue:** `config_phase1.py` không có cơ chế riêng cho `metric_for_best` khi `task_type='emotion'`. Hiện tại cả sentiment và emotion đều dùng `mae` làm metric chọn checkpoint tốt nhất. Với emotion, nên dùng `mean_f1` thay vì `mae`.

---

### 2.2. Phản biện 2: Khoảng cách Cross-lingual

| Khía cạnh | Review đề xuất | Code thực tế | Xác minh |
|:---|:---|:---|:---|
| Dataset tiếng Việt | `aligned_50_vi.pkl` dùng PhoBERT | GCS bucket **không có** file này | ✅ **Chưa cần fix** — future work |
| Cross-lingual gap | Cần Aligner Layer hoặc fine-tune text encoder | Chưa implement | ✅ **Chưa cần fix** — future work |

**Kết luận:** Vấn đề cross-lingual chưa cần giải quyết ở giai đoạn này. Chỉ cần khi nào `aligned_50_vi.pkl` được tạo và upload lên GCS.

---

### 2.3. Phản biện 3: Nguy cơ OOM khi d_model=128

| Khía cạnh | Review đề xuất | Code thực tế | Xác minh |
|:---|:---|:---|:---|
| Gradient Accumulation | Cần implement để giả lập batch_size lớn | **Chưa có** trong `trainer.py` | 🔴 **CẦN FIX** |
| batch_size cho unaligned | Nên giảm xuống 16 hoặc 8 | Notebook 04 chỉ có batch_size=16 | ✅ **Đúng** |
| Cross-attention O(n²) | seq_len=500 gây tăng quadratically | Hiện notebook 04 chưa train với d_model=128 | ⚠️ **Cần theo dõi** |

---

## 3. Issues mới phát hiện ngoài review

### 3.1. 🔴 CRITICAL — Emotion CSV Logging Dispatch Bug

**Mức độ:** Nghiêm trọng — emotion metrics bị silent drop

**Vấn đề:** Trong `trainer.py`, hàm `evaluate()` dispatch đúng sang `compute_emotion_metrics()` khi `task_type='emotion'`, nhưng `evaluate_and_save()` và `_append_history()` luôn dùng `metrics_to_row()` và sentiment fieldnames.

**Code hiện tại:**

```python
# trainer.py:300-304 — evaluate() dispatch ĐÚNG
if self.task_type == "emotion":
    metrics = compute_emotion_metrics(y_true, y_pred)  # ← return dict với mean_f1, happy_f1...
else:
    metrics = compute_metrics(y_true, y_pred)

# trainer.py:308-312 — evaluate_and_save() LUÔN dùng sentiment formatter
def evaluate_and_save(self, data_loader, split='test', epoch=0):
    loss, metrics = self.evaluate(data_loader, split=split, epoch=epoch)
    row = metrics_to_row(split, epoch, loss, metrics)  # ← SAI: luôn dùng sentiment
    self._append_history([row])

# trainer.py:358-368 — fieldnames chỉ có sentiment fields
fieldnames = ["split", "epoch", "loss", "mae", "mse", "corr",
              "acc2", "acc5", "acc7", "f1", "train_step_loss"]
# ← Không có: mean_f1, mean_acc, mean_mae, happy_f1, sad_f1...
```

**Hệ quả:** Khi train emotion task:
- Metrics được compute đúng (`mean_f1`, `happy_f1`, v.v.)
- Nhưng khi ghi CSV: tất cả emotion fields đều là `""` (empty)
- Không crash nhưng metrics bị **silent drop hoàn toàn**
- WandB logging vẫn work (vì dùng dict key trực tiếp)

**Fix cần thiết:**

```python
# trainer.py — evaluate_and_save()
def evaluate_and_save(self, data_loader, split='test', epoch=0):
    loss, metrics = self.evaluate(data_loader, split=split, epoch=epoch)
    if self.task_type == "emotion":
        row = emotion_metrics_to_row(split, epoch, loss, metrics)
    else:
        row = metrics_to_row(split, epoch, loss, metrics)
    self._append_history([row])
    return row
```

```python
# trainer.py — _append_history() — thêm emotion fieldnames
EMOTION_FIELDNAMES = [
    "split", "epoch", "loss",
    "mean_f1", "mean_acc", "mean_mae",
    "happy_f1", "sad_f1", "angry_f1",
    "surprise_f1", "disgust_f1", "fear_f1",
]

SENTIMENT_FIELDNAMES = [
    "split", "epoch", "loss", "train_step_loss",
    "mae", "mse", "corr", "acc2", "acc5", "acc7", "f1",
]
```

---

### 3.2. 🔴 CRITICAL — GitHub Repo Trống

**Mức độ:** Nghiêm trọng — notebook sẽ crash khi import

**Vấn đề:** Notebook cell 2 clone repo:

```python
REPO_URL = 'https://github.com/Kandesfx/Training-Multimodal-Emotion-Analysis.git'
get_ipython().system(f'git clone {REPO_URL} {REPO_PATH}')
```

Repository tồn tại nhưng **0 code bên trong** (0 stars, 0 forks). Khi clone về:
- Thư mục `/content/BCDA` được tạo nhưng **trống**
- `sys.path.append(str(REPO_PATH))` không có gì để import
- Tất cả import (`training.config_phase1`, `training.dataset_mosei`, v.v.) sẽ **ModuleNotFoundError**

**Guard hiện tại không hoạt động:**
```python
if '<YOUR_REPO_URL_HERE>' in REPO_URL:  # ← URL đã được set, không raise
    raise ValueError('Hãy thay REPO_URL...')
```

**Giải pháp:**
1. Push toàn bộ project lên GitHub repo (bao gồm `training/`, `scripts/`, `notebooks/`, `tools/`, v.v.)
2. Hoặc thay đổi `REPO_SOURCE = 'drive'` và dùng Google Drive thay vì GitHub

---

### 3.3. 🟡 MEDIUM — GCS Download vs Upload Inconsistent

**Mức độ:** Medium — có thể gây lỗi GCS download

**Vấn đề:** `trainer.py` upload dùng `gsutil` đúng cách:
```python
subprocess.run(["gsutil", "cp", str(local_path), gcs_dest], ...)
```

Nhưng notebook cell 2 download dùng `gcloud`:
```python
get_ipython().system(f'gcloud storage cp gs://{GCS_BUCKET}/...')
```

`gcloud storage cp` cần thêm config:
1. `gcloud config set project <project-id>` — **thiếu**
2. Hoặc dùng `!gsutil cp` thay vì `gcloud storage cp` (đơn giản hơn, Colab pre-installed)

**Fix đề xuất:**
```python
# Thay:
get_ipython().system(f'gcloud storage cp gs://{GCS_BUCKET}/...')
# Bằng:
get_ipython().system(f'!gsutil cp gs://{GCS_BUCKET}/...')
# Hoặc dùng Python API:
from google.cloud import storage
client = storage.Client()
bucket = client.bucket(GCS_BUCKET)
blob = bucket.blob('data/MSA-Dataset/aligned_50.pkl')
blob.download_to_filename('/content/data/MSA-Dataset/aligned_50.pkl')
```

---

### 3.4. 🟡 MEDIUM — Metric for Best khi Emotion Task

**Mức độ:** Medium — checkpoint selection không tối ưu cho emotion

**Vấn đề:** `config_phase1.py` có:
```python
metric_for_best: str = "mae"
maximize_metric: bool = False
```

Điều này áp dụng cho **cả** sentiment và emotion task. Nhưng emotion task nên dùng:
```python
metric_for_best: str = "mean_f1"
maximize_metric: bool = True
```

**Fix đề xuất:** Thêm dynamic config trong `Phase1Trainer.__init__()`:
```python
if self.task_type == "emotion":
    self.metric_for_best = "mean_f1"
    self.maximize_metric = True
else:
    self.metric_for_best = self.config.training.metric_for_best
    self.maximize_metric = self.config.training.maximize_metric
```

---

### 3.5. 🟡 MEDIUM — Gradient Accumulation chưa có

**Mức độ:** Medium — risk OOM khi unaligned + d_model=128

**Vấn đề:** `trainer.py` không có gradient accumulation. Khi chạy unaligned (seq_len=500) với d_model=128 và batch_size=16 trên GPU 15GB, có risk OOM.

**Fix đề xuất:** Thêm vào `Phase1TrainingConfig`:
```python
gradient_accumulation_steps: int = 1  # mặc định = 1 (không thay đổi)
effective_batch_size: int = 32        # batch_size * gradient_accumulation_steps
```

Và trong `_run_epoch()`:
```python
if training:
    self.optimizer.zero_grad(set_to_none=True)

for step, batch in enumerate(data_loader):
    # ... forward ...
    loss = self.criterion(preds, labels) / self.config.training.gradient_accumulation_steps
    self.scaler.scale(loss).backward()

    if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(...)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
```

---

### 3.6. 🟢 MINOR — Evaluator Emotion MAE Formula

**Mức độ:** Minor — metric phụ, không ảnh hưởng training

**Vấn đề:** `evaluator_emotion.py:92`:
```python
mae_per_emo = np.mean(np.abs(y_true - y_pred_prob * 3.0), axis=0)
```

- `y_pred_prob` là sigmoid output (0,1) × 3.0 → (0,3)
- `y_true` là intensity (0,3)
- Đây là xấp xỉ tuyến tính — không hoàn toàn chính xác về mặt toán học (review đã chỉ ra)

**Nhận xét:** MAE trong emotion chỉ là supplementary metric. Không ảnh hưởng training hay model selection. Không cần fix ngay.

---

## 4. Danh sách fix theo ưu tiên

```
┌─────────────────────────────────────────────────────────────────┐
│  PRIORITY 1 — Trước khi train lần đầu                          │
├─────────────────────────────────────────────────────────────────┤
│  🔴 [E1] Fix emotion CSV dispatch (trainer.py:310)              │
│       → Emotion metrics được ghi đúng vào CSV                   │
│                                                                 │
│  🔴 [E2] Push code lên GitHub repo                              │
│       → Hoặc đổi REPO_SOURCE='drive'                           │
│       → Không thì notebook sẽ crash ngay khi chạy             │
│                                                                 │
│  🟡 [E3] Fix GCS download (notebook cell 2)                    │
│       → Thêm `gcloud config set project` hoặc dùng `gsutil`    │
│       → Tránh lỗi authentication trên Colab                     │
│                                                                 │
│  🟡 [E4] Dynamic metric_for_best cho emotion task                │
│       → mean_f1 thay vì mae khi task_type='emotion'            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  PRIORITY 2 — Sau khi baseline chạy ổn định                    │
├─────────────────────────────────────────────────────────────────┤
│  🟡 [E5] Gradient Accumulation (trainer.py)                    │
│       → Phòng OOM khi unaligned + d_model=128                   │
│                                                                 │
│  🟢 [E6] Cập nhật document COMPREHENSIVE_TRAINING_STRATEGY     │
│       → Thêm OOM warning cho unaligned + d_model=128            │
│       → Thêm metric_for_best recommendation cho emotion         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  PRIORITY 3 — Future work (khi có aligned_50_vi.pkl)            │
├─────────────────────────────────────────────────────────────────┤
│  🟢 [E7] Cross-lingual Aligner Layer cho Vietnamese branch      │
│       → PhoBERT → BERT space projection                         │
│       → Fine-tune text encoder thay vì freeze                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Kết luận

### Tổng quan

| Điểm | Số lượng |
|:---|:---:|
| Review đúng với code | 5 |
| Cần fix trước khi train | 2 🔴 + 2 🟡 |
| Future work | 2 🟢 |
| Không cần fix | 1 🟢 |

### Đánh giá chiến lược

Chiến lược trong `COMPREHENSIVE_TRAINING_STRATEGY.md` **đúng về cơ bản**:
- ✅ Config P0+P1 đã được apply đúng
- ✅ Dataset pipeline chính xác
- ✅ MulT architecture đúng
- ✅ Training loop (AMP, gradient clipping, cosine warmup) đúng
- ✅ Sentiment metrics đúng
- ⚠️ Emotion metrics cần fix dispatch

### Hành động tiếp theo

**Trước khi chạy notebook trên Colab:**
1. Fix `trainer.py:310` — emotion CSV dispatch
2. Push code lên GitHub HOẶC đổi `REPO_SOURCE='drive'`
3. Fix GCS download command

**Sau khi baseline chạy ổn định:**
4. Thêm gradient accumulation cho unaligned training
5. Dynamic `metric_for_best` cho emotion task
