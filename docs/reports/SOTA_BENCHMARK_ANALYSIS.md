# PHỤ LỤC: SOTA BENCHMARK ANALYSIS — CMU-MOSEI EMOTION RECOGNITION

**Đề tài:** Đề Tài 17 — Hệ Thống Phân Tích Cảm Xúc Đa Phương Thức
**Ngày:** 09/06/2026
**Nguồn:** Tổng hợp từ ACL Anthology, ArXiv, Semantic Scholar

---

## Mục lục

1. [Bối cảnh — Tại sao bài toán này khó?](#1-bối-cảnh--tại-sao-bài-toán-này-khó)
2. [Community SOTA trên CMU-MOSEI Emotion](#2-community-sota-trên-cmu-mosei-emotion)
3. [So sánh với Model hiện tại (Phase 1)](#3-so-sánh-với-model-hiện-tại-phase-1)
4. [Aligned vs Unaligned — Dataset Comparison](#4-aligned-vs-unaligned--dataset-comparison)
5. [Đặt kỳ vọng thực tế cho Phase 2](#5-đặt-kỳ-vọng-thực-tế-cho-phase-2)

---

## 1. Bối cảnh — Tại sao bài toán này khó?

### 1.1. Class Imbalance nghiêm trọng

CMU-MOSEI có phân bố 6 emotions cực kỳ mất cân bằng:

| Emotion | Positive Rate | Imbalance Ratio | Chiếm % dataset |
|:---|:---:|:---:|:---:|
| Happy | 34.0% | 1.9:1 | 65.9% của tất cả positive |
| Sad | 14.1% | 6.1:1 | — |
| Angry | 14.5% | 5.9:1 | — |
| Disgust | 11.9% | 7.4:1 | — |
| Surprise | 3.3% | 29.3:1 | — |
| Fear | 2.2% | 44.4:1 | — |

```
→ Happy chiếm 65.9% tất cả positive labels
→ Fear/Surprise/Disgust chỉ chiếm <7% còn lại
→ Model tối ưu hóa accuracy đơn thuần sẽ luôn predict Happy
```

### 1.2. Multi-label complexity

CMU-MOSEI cho phép multiple emotions đồng thời trong một utterance. Người nói có thể vừa Happy vừa Surprise, hoặc Sad và Angry. Điều này làm phức tạp thêm bài toán.

### 1.3. Đây là vấn đề CỦA CẢ CỘNG ĐỒNG NGHIÊN CỨU

Ngay cả các mô hình SOTA (2024-2025) với hàng triệu parameters vẫn gặp khó khăn nghiêm trọng với Fear/Surprise/Disgust trên MOSEI.

---

## 2. Community SOTA trên CMU-MOSEI Emotion

### 2.1. Emotion Classification Results (6-class, Aligned data)

Nguồn: Zadeh et al. (2018), Tsai et al. (2019), Affect-Diff (2025), LDDU (2025)

| Paper / Model | Year | Weighted Acc | Micro F1 | Happy | Sad | Angry | Fear | Disgust | Surprise |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **MulT (paper)** | 2019 | — | — | 0.784 | 0.340 | 0.231 | **0.000** | **0.000** | **0.000** |
| TFN | 2018 | — | — | 0.807 | 0.362 | 0.080 | **0.000** | **0.000** | **0.000** |
| MISA | 2020 | — | — | 0.789 | 0.428 | 0.111 | **0.000** | **0.000** | **0.000** |
| MMIM | 2021 | — | — | 0.808 | 0.422 | 0.262 | **0.000** | **0.000** | **0.000** |
| TETFN | 2022 | — | — | 0.768 | 0.368 | 0.167 | **0.000** | **0.000** | **0.000** |
| **Affect-Diff** | 2025 | **38.4%** | 0.214 | 0.734 | 0.375 | 0.175 | **0.000** | **0.000** | **0.000** |
| **LDDU** | 2025 | — | **+4.3%** | — | — | — | — | — | — |
| MTL (STL) | 2019 | 60.8% | — | — | — | — | — | — | — |
| MTL (MTL) | 2019 | 62.8% | 78.6% | — | — | — | — | — | — |

> **Nhận xét:** Affect-Diff (2025) sử dụng Causal Graph + Diffusion Prior + VAE bottleneck — kiến trúc phức tạp nhất — nhưng Fear/Disgust/Surprise vẫn = 0.000 F1.

### 2.2. So sánh Sentiment vs Emotion

| Task | Metric | SOTA Value | Model | Year |
|:---|:---|:---:|:---|:---:|
| Sentiment Regression | MAE | ~0.478 | Various MLLMs | 2024-2025 |
| Sentiment Classification | Acc-2 | ~88.5% | EmoVerse-8B | 2024 |
| Emotion (6-class) | Micro F1 | ~0.214-0.35 | Various | 2018-2025 |
| Emotion (6-class) | Weighted Acc | ~38-63% | Various | 2018-2025 |

```
→ Emotion recognition KHÓ HƠN 10x so với Sentiment trên cùng dataset
→ Micro F1 ~0.214 là CEILING thực tế cho bài toán này
→ Mean F1 (trung bình 6 emotions) còn thấp hơn nữa
```

### 2.3. Các approaches đã thử của cộng đồng

| Approach | Kết quả | Ghi chú |
|:---|:---:|:---|
| BCE Loss thuần | Fear=0 | Không handle imbalance |
| BCE + pos_weight | Fear tăng nhẹ | Vẫn không đủ |
| Focal Loss | Tốt hơn BCE | Giảm penalty cho easy negatives |
| Weighted Sampling | Tăng recall rare | Có thể giảm precision |
| Multi-task Learning | ~+2% W-Acc | Chia sẻ với sentiment |
| Causal Graph | +18% balanced acc | Affect-Diff |
| Diffusion Prior | +24% (ablation) | Chỉ với VAE variant |
| Pre-trained MLLMs | Acc-2 ~82-88% | Sentiment, không phải emotion |

---

## 3. So sánh với Model hiện tại (Phase 1)

### 3.1. Performance Gap Analysis

| Metric | MulT P0 (Phase 1) | MulT Baseline 2018 | Gap | Assessment |
|:---|:---:|:---:|:---:|:---|
| Happy F1 | 0.4709 | 0.784 | -40% | Dưới baseline |
| Sad F1 | ~0.27 (est) | 0.340 | -21% | Dưới baseline |
| Angry F1 | ~0.19 (est) | 0.231 | -18% | Dưới baseline |
| Fear F1 | 0.0348 | 0.000 | N/A | Vượt baseline (!) |
| Mean F1 | 0.2064 | ~0.21 (est) | ~0% | Tương đương |

```
→ Model hiện tại đang ở MỨC BASELINE CỦA CỘNG ĐỒNG 2018
→ Happy F1 thấp nhất trong class phổ biến (có thể do loss function)
→ Fear F1 = 0.0348 > 0.000 của TFN/MulT gốc (may do threshold)
```

### 3.2. Root cause từ góc nhìn SOTA

Các root causes đã xác định trong Phase 1 đều được xác nhận bởi SOTA literature:

1. **BCE without pos_weight** → cộng đồng đã xác nhận cần weighted loss hoặc sampling
2. **Fixed threshold 0.5** → SOTA papers đều dùng per-class threshold hoặc calibration
3. **d_model=64 bottleneck** → cần lớn hơn, nhưng SOTA vẫn chưa giải quyết được rare classes
4. **Không có weighted sampling** → ảnh hưởng đến gradient direction

---

## 4. Aligned vs Unaligned — Dataset Comparison

### 4.1. Dataset Specifications

| Property | Aligned (`aligned_50.pkl`) | Unaligned (`unaligned_50.pkl`) |
|:---|:---|:---|
| Text sequence | 50 timesteps | 50 timesteps |
| Audio sequence | 50 timesteps | 500 timesteps |
| Vision sequence | 50 timesteps | 500 timesteps |
| Temporal alignment | ✅ Perfect sync | ❌ No sync |
| Total samples | 22,856 | ~23,453 |
| Train samples | 16,326 | ~16,000 |
| Class imbalance | 15.5:1 (Happy:Fear) | 15.5:1 (same) |

### 4.2. Nên dùng dataset nào?

**Sentiment Regression → Unaligned tốt hơn:**
- 500 timesteps audio/vision → model thấy nhiều frames hơn
- Temporal coverage tốt hơn cho regression task

**Emotion Classification → Aligned tốt hơn (ưu tiên):**
- Temporal alignment quan trọng để biết cảm xúc xảy ra ở đâu
- Cross-modal attention tự học alignment tốt hơn khi có ground truth alignment
- Unaligned vẫn dùng được nếu model đủ mạnh

**Tối ưu nhất:** Ensemble cả 2 models (1 trên aligned, 1 trên unaligned)

### 4.3. MulT Cross-Modal Attention trên Unaligned

```
Aligned:    Q(text)=50, K(audio)=50, K(vision)=50
            → Cross-modal attention đồng kích thước

Unaligned:  Q(text)=50, K(audio)=500, K(vision)=500
            → Q=50, K/V=500: model thấy 10x audio/vision frames
            → Attention scores chọn relevant frames từ 500
            → Better temporal coverage cho audio/vision
```

---

## 5. Đặt kỳ vọng thực tế cho Phase 2

### 5.1. Target Metrics

| Metric | Current (P0) | Target Round 2 | SOTA Ceiling | Notes |
|:---|:---:|:---:|:---:|:---|
| Happy F1 | 0.4709 | 0.55-0.65 | ~0.80 | Dễ cải thiện nhất |
| Sad F1 | ~0.27 | 0.35-0.42 | ~0.43 | |
| Angry F1 | ~0.19 | 0.28-0.35 | ~0.26 | |
| Disgust F1 | ~0.17 | 0.22-0.30 | ~0.15 | SOTA ~0.15 |
| Surprise F1 | ~0.10 | 0.15-0.25 | ~0.10 | SOTA ~0.10 |
| Fear F1 | ~0.035 | 0.08-0.15 | ~0.05 | SOTA ~0.00-0.125 |
| **Mean F1** | **0.2064** | **0.28-0.38** | **~0.35-0.45** | Primary metric |
| Weighted Acc | ~25% | 35-45% | ~40-63% | Balanced metric |

### 5.2. Khi nào thì "đạt"?

```
Không có ngưỡng "pass/fail" cố định. Các mốc tham chiếu:

✅ THÀNH CÔNG (Minimum viable):
  - Mean F1 >= 0.25
  - Fear F1 > 0.05 (bất kỳ model nào đều thấp)
  - Weighted Acc >= 30%

✅ TỐT:
  - Mean F1 >= 0.30
  - Weighted Acc >= 35%

✅ XUẤT SẮC:
  - Mean F1 >= 0.35
  - Happy F1 >= 0.55
  - Weighted Acc >= 40%

⚠️ KHÔNG ĐẠT:
  - Mean F1 < 0.20 (dưới baseline)
  - Happy F1 giảm rõ ràng (loss function có vấn đề)
```

### 5.3. Chiến lược để tối đa hóa kết quả

```
Round 2A: Quick wins (không cần train lại)
  → Per-emotion threshold tuning
  → Expected: +5-15% Mean F1

Round 2B: Primary training (MulT P1 + Focal Loss)
  → d_model=128, BCE pos_weight, Focal Loss
  → Expected: +30-80% Mean F1

Round 2C: Sentiment baseline (MulT P1 cho sentiment)
  → New baseline để so sánh với Improved LSTM

Long-term (Phase 2):
  → Pseudo-labeling cho Fear/Surprise từ unaligned data
  → Weighted sampling
  → Cross-lingual transfer learning (sau khi có Vietnamese data)
```

---

*Nguồn: CMU-MOSEI paper (Zadeh et al., ACL 2018), MulT paper (Tsai et al., ACL 2019), Affect-Diff (ArXiv 2025), LDDU (ACL Findings 2025), M2SE/EmoVerse (ArXiv 2024), MOSEI benchmark (OpenCodePapers)*
