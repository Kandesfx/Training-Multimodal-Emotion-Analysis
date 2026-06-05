# 🎯 Phương Pháp Thu Thập Dữ Liệu Tiếng Việt — Tối Ưu Năng Suất & Tự Động Hóa

## Đề Tài 17: Hệ Thống Phân Tích Cảm Xúc Đa Phương Thức

Tài liệu này nghiên cứu các phương pháp thu thập (cào) dữ liệu video tiếng Việt hiệu quả nhất, phân tích các nguồn dữ liệu tiềm năng, và đề xuất quy trình tự động hóa tối đa pipeline của EDS tool.

---

## Mục Lục

1. [Phân Tích Yêu Cầu Dữ Liệu](#1-phân-tích-yêu-cầu-dữ-liệu)
2. [Nguồn Dữ Liệu Tiếng Việt Tiềm Năng](#2-nguồn-dữ-liệu-tiếng-việt-tiềm-năng)
3. [Chiến Lược Thu Thập Theo Loại Cảm Xúc](#3-chiến-lược-thu-thập-theo-loại-cảm-xúc)
4. [Khả Năng Tự Động Hóa Hiện Tại của EDS Tool](#4-khả-năng-tự-động-hóa-hiện-tại-của-eds-tool)
5. [Đề Xuất Nâng Cấp — Chế Độ Batch Harvest](#5-đề-xuất-nâng-cấp-chế-độ-batch-harvest)
6. [Kịch Bản Thu Thập Tối Ưu (End-to-End Workflow)](#6-kịch-bản-thu-thập-tối-ưu)
7. [Ước Tính Thời Gian & Năng Suất](#7-ước-tính-thời-gian-và-năng-suất)
8. [Kiểm Soát Chất Lượng & Lọc Dữ Liệu Xấu](#8-kiểm-soát-chất-lượng)
9. [Xử Lý Rào Cản Kỹ Thuật](#9-xử-lý-rào-cản-kỹ-thuật)

---

## 1. Phân Tích Yêu Cầu Dữ Liệu

### 1.1. Mục Tiêu Số Lượng

Từ phân tích trong [TRAINING_ROADMAP.md](file:///d:/Hai/study/DeepLerning/BCDA/docs/research/TRAINING_ROADMAP.md), mô hình cần tối thiểu **2,000–5,000 clips** tiếng Việt đã gán nhãn để Fine-tuning đạt hiệu quả:

| Kịch bản | Số clips cần | Số video gốc ước tính | Thời gian video ước tính |
|:---|:---|:---|:---|
| Tối thiểu (khả thi) | 2,000 | ~100–200 video | ~50–100 giờ |
| Khuyên dùng | 3,000–5,000 | ~200–400 video | ~100–200 giờ |
| Lý tưởng | 5,000–10,000 | ~400–700 video | ~200–350 giờ |

*Ước tính dựa trên tỷ lệ trung bình: 1 video YouTube dài ~30 phút → EDS tạo ra ~15–25 clips hợp lệ (có mặt người + lời thoại).*

### 1.2. Tiêu Chí Chất Lượng Một Clip Hợp Lệ

Để clip có thể dùng cho training mô hình đa phương thức, nó **phải đồng thời** đáp ứng:

| Tiêu chí | Yêu cầu | Lý do |
|:---|:---|:---|
| **Khuôn mặt rõ** | ≥ 1 khuôn mặt nhìn thấy trong ≥ 70% thời lượng clip | Nhánh Vision cần Action Units liên tục |
| **Lời thoại** | Có transcript tiếng Việt ≥ 3 từ | Nhánh Text cần embedding PhoBERT |
| **Âm thanh sạch** | Không nhiễu nặng (nhạc nền lấn át, tiếng ồn) | Nhánh Audio cần đặc trưng COVAREP rõ |
| **Cảm xúc rõ ràng** | Biểu cảm/giọng nói thể hiện cảm xúc phân biệt được | Nhãn sentiment phải có ý nghĩa |
| **Độ dài** | 3–15 giây (tối ưu 5–8 giây) | Đủ ngắn để gán nhãn, đủ dài để có ngữ cảnh |
| **1 người nói chính** | Ưu tiên clip chỉ có 1 người nói đang thể hiện cảm xúc | Tránh nhầm lẫn cảm xúc giữa 2 người |

---

## 2. Nguồn Dữ Liệu Tiếng Việt Tiềm Năng

### 2.1. Bảng Đánh Giá Nguồn

| Nguồn | Loại nội dung | Cảm xúc mạnh? | Chất lượng hình/âm | Độ dài phù hợp? | Dễ cào? | ★ Điểm tổng |
|:---|:---|:---|:---|:---|:---|:---|
| **Phim Việt Nam (YouTube)** | Drama, tình cảm, hài | ✅ Rất mạnh | ✅ Cao (HD) | ✅ Clip 5–10p dễ cắt | ✅ Nhiều trên YT | ⭐⭐⭐⭐⭐ |
| **Talkshow / Phỏng vấn** | Trò chuyện, chia sẻ | ⚠️ Trung bình | ✅ Cao | ✅ Hội thoại rõ | ✅ Nhiều trên YT | ⭐⭐⭐⭐ |
| **Phóng sự / Tin tức** | Báo cáo, phản ánh | ⚠️ Ít đa dạng (chủ yếu neutral/sad) | ✅ Cao | ⚠️ Ít hội thoại cảm xúc | ✅ Nhiều | ⭐⭐⭐ |
| **Vlog / Review phim** | Đánh giá, bình luận cá nhân | ✅ Mạnh (happy/angry/disgust) | ⚠️ Trung bình | ✅ 1 người nói rõ | ✅ Rất nhiều | ⭐⭐⭐⭐⭐ |
| **Gameshow / Reality TV** | Tranh tài, thử thách | ✅ Rất mạnh (surprise/happy/angry) | ✅ Cao | ✅ Phản ứng ngắn | ✅ Nhiều | ⭐⭐⭐⭐⭐ |
| **TikTok / Shorts** | Clip ngắn đa dạng | ✅ Mạnh | ⚠️ Biến động | ⚠️ Quá ngắn đôi khi | ⚠️ Khó cào hàng loạt | ⭐⭐⭐ |
| **Podcast tiếng Việt** | Hội thoại dài | ⚠️ Chậm, ít biểu cảm mặt | ⚠️ Thường chỉ audio | ❌ Không có video mặt | ❌ Không phù hợp | ⭐ |

### 2.2. Top 3 Nguồn Khuyên Dùng

#### 🥇 Nguồn 1: Phim Truyền Hình / Web Drama Việt Nam (YouTube)

**Tại sao hiệu quả nhất:**
- Diễn viên chuyên nghiệp → biểu cảm khuôn mặt **rõ ràng và đa dạng**
- Chất lượng quay phim/ánh sáng **đạt chuẩn studio** → không bị nhiễu visual
- Âm thanh thu phòng → **ít tạp âm** → trích xuất COVAREP chính xác
- Lời thoại tiếng Việt chuẩn → **Whisper nhận diện tốt**
- 1 tập phim 40-60 phút → EDS cắt ra **20-40 clips hợp lệ**

**Từ khóa tìm kiếm YouTube:**
```
- "phim truyền hình Việt Nam 2024 tập [số]"
- "web drama Việt Nam hay nhất"
- "[Tên phim] tập [số] full"
- "phim Việt Nam cảm động"
- "phim hài Việt Nam" (cho nhãn happy)
- "phim tâm lý Việt Nam" (cho nhãn sad/fear)
```

**Kênh YouTube giàu nội dung:**
- VTV Giải Trí, HTV Phim, THVL Giải Trí, Galaxy Play
- Vie Channel (The Face, Người Ấy Là Ai)
- SCHANNEL (phim ngắn cảm xúc mạnh)

#### 🥈 Nguồn 2: Vlog / Review Phim / Reaction Videos

**Tại sao hiệu quả:**
- Người nói **nhìn thẳng camera** → khuôn mặt luôn rõ, dễ phát hiện AUs
- **1 người nói duy nhất** → không nhầm lẫn cảm xúc
- Biểu cảm **tự nhiên** (không diễn) → phản ánh đúng cảm xúc thực tế
- Nội dung đa dạng: vui, tức giận, thất vọng, bất ngờ, ghê tởm

**Từ khóa:**
```
- "review phim [tên phim] reaction"
- "thử ăn [món ăn] reaction" (disgust/surprise)
- "vlog Việt Nam cuộc sống" (neutral/happy)
- "chia sẻ chuyện buồn" (sad)
- "phản ứng khi biết [sự kiện]" (surprise/fear)
```

#### 🥉 Nguồn 3: Gameshow / Reality Show Việt Nam

**Tại sao hiệu quả:**
- Phản ứng cảm xúc **cực kỳ mạnh và đa dạng** (thắng/thua/bất ngờ/sợ hãi)
- Camera quay cận mặt người chơi → **khuôn mặt lớn, rõ nét**
- Chất lượng sản xuất cao (HD/4K)
- Âm thanh tốt (có mic riêng cho mỗi người chơi)

**Chương trình gợi ý:**
```
- "Người Ấy Là Ai" → surprise, happy, sad
- "Running Man Vietnam" → happy, fear, surprise
- "Thách Thức Danh Hài" → happy, surprise
- "The Voice Vietnam" → happy, sad, surprise
- "Rap Việt" → happy, angry (battle), surprise
- "2 Ngày 1 Đêm" → happy, surprise, fear
```

---

## 3. Chiến Lược Thu Thập Theo Loại Cảm Xúc

### 3.1. Phân Bổ Mục Tiêu

Để tránh mất cân bằng nhãn (class imbalance), cần thu thập **đều** cho mỗi cảm xúc:

| Cảm xúc | % mục tiêu | Số clips mục tiêu (tổng 3000) | Nguồn chính | Ghi chú |
|:---|:---|:---|:---|:---|
| **Happy** | 20% | 600 | Hài, gameshow, vlog tích cực | Dễ thu thập nhất |
| **Sad** | 15% | 450 | Drama tình cảm, chia sẻ buồn | Dễ thu thập |
| **Angry** | 15% | 450 | Drama xung đột, tranh cãi | Trung bình |
| **Neutral** | 15% | 450 | Tin tức, phỏng vấn, hướng dẫn | Dễ nhưng ít giá trị |
| **Surprise** | 15% | 450 | Gameshow, reaction, twist phim | Trung bình |
| **Fear** | 10% | 300 | Phim kinh dị, thử thách | Khó thu thập hơn |
| **Disgust** | 10% | 300 | Thử ăn, review đồ ăn, phim | Khó thu thập nhất |

### 3.2. Chiến Thuật "Nhắm Cảm Xúc" Khi Chọn Video

Thay vì cào ngẫu nhiên rồi hy vọng có đủ cảm xúc, hãy **nhắm mục tiêu** từ đầu:

```
  Thu thập theo đợt (batch):
  
  Đợt 1 (Tuần 1): Happy + Surprise
    → Cào 50 video gameshow/hài → ~600 clips happy + 300 clips surprise
    
  Đợt 2 (Tuần 2): Sad + Angry
    → Cào 50 video drama tình cảm + xung đột → ~400 clips sad + 400 clips angry
    
  Đợt 3 (Tuần 3): Fear + Disgust + Neutral
    → Cào 30 video kinh dị + review đồ ăn + tin tức → ~250 clips mỗi loại
    
  Đợt 4 (Tuần 4): Bổ sung các nhãn thiếu
    → Kiểm tra dashboard thống kê → cào thêm cho các nhãn chưa đủ quota
```

---

## 4. Khả Năng Tự Động Hóa Hiện Tại của EDS Tool

### 4.1. Những Gì EDS Đã Làm Tự Động

Phân tích mã nguồn hiện tại, EDS tool đã có pipeline tự động hóa khá mạnh:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Đã tự động (chỉ cần nhập 1 URL YouTube):                      │
  │                                                                 │
  │  ✅ Tải video (yt-dlp, 3 profile fallback, cookie support)     │
  │  ✅ Phát hiện chuyển cảnh (PySceneDetect)                      │
  │  ✅ Cắt clip thông minh (SmartSegmenter: face + dialogue)      │
  │  ✅ Phát hiện khuôn mặt (MTCNN + tracking IoU)                │
  │  ✅ Tách âm thanh (FFmpeg → .wav 16kHz)                        │
  │  ✅ Nhận diện lời thoại (Whisper tiếng Việt)                   │
  │  ✅ Phân loại cảm xúc (DeepFace + Lexicon + Wav2Vec2)         │
  │  ✅ Chấm điểm chất lượng (QualityScorer)                      │
  │  ✅ Tự động duyệt (auto_approved nếu quality ≥ 0.80)          │
  │                                                                 │
  │  Tổng: 1 click "Process" → toàn bộ pipeline chạy tự động       │
  └─────────────────────────────────────────────────────────────────┘
```

### 4.2. Những Gì CẦN Làm Thủ Công (Nút Thắt Cổ Chai)

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  Phải làm thủ công (tốn thời gian nhất):                       │
  │                                                                 │
  │  ❌ Nhập URL từng video một → CẦN: Batch import danh sách URL │
  │  ❌ Bấm "Process" cho từng video → CẦN: Auto-queue processing │
  │  ❌ Duyệt/sửa nhãn từng clip → CẦN: Smart auto-approve       │
  │  ❌ Không tự tìm kiếm video mới → CẦN: Keyword search engine  │
  │  ❌ Không có thống kê phân bổ cảm xúc theo thời gian thực      │
  │     → CẦN: Dashboard target tracking                           │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 5. Đề Xuất Nâng Cấp — Chế Độ Batch Harvest

### 5.1. Tính Năng #1: Batch URL Import

**Hiện tại:** Phải nhập từng URL một qua giao diện.
**Nâng cấp:** Cho phép paste danh sách URL hoặc import file `.txt`:

```
  File: harvest_urls.txt
  ─────────────────────
  https://www.youtube.com/watch?v=abc123  # Phim hài Việt
  https://www.youtube.com/watch?v=def456  # Gameshow
  https://www.youtube.com/watch?v=ghi789  # Drama tình cảm
  ...
```

**Cách triển khai trong API:**
```python
# Endpoint mới: POST /videos/batch
@router.post("/batch")
def create_batch_videos(urls: List[str], db: Session = Depends(get_db)):
    """Import hàng loạt URL và tự động xếp hàng xử lý."""
    created = []
    for url in urls:
        video = Video(title="Pending...", source_url=url, status="queued")
        db.add(video)
        created.append(video)
    db.commit()
    return {"imported": len(created), "videos": created}
```

### 5.2. Tính Năng #2: Auto-Queue Processing

**Hiện tại:** Phải bấm "Process" thủ công cho từng video.
**Nâng cấp:** Hệ thống tự động chạy pipeline lần lượt cho các video ở trạng thái `queued`:

```
  Auto-Queue Worker Flow:
  
  while True:
      video = db.query(Video).filter(status == "queued").first()
      if video:
          video.status = "processing"
          orchestrator.run_pipeline(video.id, db)
          video.status = "completed"
      else:
          sleep(30)  # Chờ 30 giây rồi kiểm tra lại
```

**Lợi ích:** Người dùng chỉ cần paste 50 URL → đi ngủ → sáng hôm sau tất cả đã xử lý xong.

### 5.3. Tính Năng #3: YouTube Playlist / Channel Crawler

**Hiện tại:** Phải tìm và copy URL từng video thủ công.
**Nâng cấp:** Nhập URL playlist hoặc channel → tự động trích xuất tất cả video URLs:

```python
# Yt-dlp đã hỗ trợ sẵn playlist extraction
def extract_playlist_urls(playlist_url: str, max_videos: int = 50) -> List[str]:
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,      # Chỉ lấy danh sách, không tải
        "playlistend": max_videos,  # Giới hạn số lượng
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        entries = info.get("entries", [])
        return [
            f"https://www.youtube.com/watch?v={entry['id']}"
            for entry in entries if entry.get("id")
        ]
```

**Ví dụ sử dụng:**
```
Input:  https://www.youtube.com/@VTVGiaiTri/videos
Output: Danh sách 50 URL video mới nhất của kênh VTV Giải Trí
→ Tự động import tất cả vào hàng đợi
```

### 5.4. Tính Năng #4: Smart Auto-Approve Nâng Cao

**Hiện tại:** QualityScorer tự động duyệt clip nếu `quality ≥ 0.80`.
**Nâng cấp — Tầng lọc bổ sung:**

```
  Bộ lọc tự động duyệt nâng cao:
  
  Clip hợp lệ ĐỂ AUTO-APPROVE nếu ĐỒNG THỜI:
  ┌─────────────────────────────────────────────────┐
  │ ✅ quality_score ≥ 0.75                         │
  │ ✅ confidence ≥ 0.65                            │
  │ ✅ agreement = "3/3" hoặc "2/3"                │
  │ ✅ has_incongruity = False                      │
  │ ✅ transcript có ≥ 3 từ tiếng Việt              │
  │ ✅ duration trong khoảng 3–15 giây              │
  │ ✅ num_faces ≥ 1 (có ít nhất 1 khuôn mặt)      │
  └─────────────────────────────────────────────────┘
  
  Clip REJECT tự động nếu BẤT KỲ:
  ┌─────────────────────────────────────────────────┐
  │ ❌ transcript rỗng hoặc < 2 từ                  │
  │ ❌ num_faces = 0 (không phát hiện khuôn mặt)    │
  │ ❌ duration < 2 giây                            │
  │ ❌ audio_clarity < 0.001 (không có âm thanh)     │
  │ ❌ predicted_emotion = "unknown"                 │
  └─────────────────────────────────────────────────┘
  
  Còn lại → needs_review (cần duyệt thủ công)
```

**Ước tính hiệu quả:** Với bộ lọc này, khoảng **50–65%** clips sẽ được tự động duyệt hoặc loại bỏ, giảm tải duyệt thủ công đáng kể.

### 5.5. Tính Năng #5: Dashboard Theo Dõi Quota Cảm Xúc

Bổ sung biểu đồ trên Dashboard hiển thị:
- Số clips đã thu thập cho mỗi cảm xúc vs mục tiêu
- Cảm xúc nào đang thiếu → gợi ý nguồn video phù hợp
- Tiến trình tổng thể (ví dụ: "2,340 / 3,000 clips — 78%")

```
  Dashboard Quota Tracking:
  
  Happy     ████████████████████░░  600/600  ✅ ĐẠT
  Sad       █████████████░░░░░░░░░  380/450  ⚠️ Thiếu 70
  Angry     ███████████░░░░░░░░░░░  320/450  ⚠️ Thiếu 130
  Neutral   ████████████████░░░░░░  450/450  ✅ ĐẠT
  Surprise  ███████████████░░░░░░░  410/450  ⚠️ Thiếu 40
  Fear      ████████░░░░░░░░░░░░░░  180/300  ❌ Thiếu 120
  Disgust   ██████░░░░░░░░░░░░░░░░  130/300  ❌ Thiếu 170
  
  Tổng: 2,470 / 3,000 (82.3%)
  
  💡 Gợi ý: Để bù "Fear" và "Disgust", hãy cào thêm:
     - Phim kinh dị Việt Nam (fear)
     - Video thử ăn côn trùng / đồ ăn lạ (disgust)
```

---

## 6. Kịch Bản Thu Thập Tối Ưu (End-to-End Workflow)

### 6.1. Quy Trình Tối Ưu Hàng Ngày

```
  ┌───────────────────────────────────────────────────────────────────┐
  │  BUỔI TỐI (30 phút thao tác → máy chạy qua đêm)               │
  │                                                                   │
  │  Bước 1: Chuẩn bị danh sách URL (15 phút)                       │
  │  → Tìm 20-30 video phù hợp trên YouTube                        │
  │  → Copy URL vào file harvest_urls.txt                            │
  │  → Nhắm mục tiêu cảm xúc đang thiếu theo Dashboard             │
  │                                                                   │
  │  Bước 2: Import và khởi chạy (5 phút)                            │
  │  → Batch import danh sách URL vào EDS                            │
  │  → Bật Auto-Queue Processing                                     │
  │  → Máy tự chạy pipeline xuyên đêm                               │
  │                                                                   │
  │  Bước 3: Sáng hôm sau — Duyệt nhanh (10-30 phút)               │
  │  → Mở Dashboard, kiểm tra kết quả                               │
  │  → Phần lớn clips đã auto-approved                              │
  │  → Chỉ duyệt clips "needs_review" (30-50 clips)                 │
  │  → Mỗi clip duyệt ~10-15 giây → ~10 phút cho 50 clips          │
  └───────────────────────────────────────────────────────────────────┘
```

### 6.2. Quy Trình Chi Tiết Cho Mỗi Đợt Thu Thập

```
  ┌─────────────────────────────────────────────────────────────────┐
  │  ĐỢT 1: HAPPY + SURPRISE (Tuần 1)                              │
  │                                                                   │
  │  Ngày 1:                                                          │
  │    Tối: Tìm 30 URL (gameshow, hài, reaction) → batch import     │
  │    Đêm: EDS chạy tự động → ~500 clips tổng, ~300 hợp lệ        │
  │    Sáng: Duyệt 50 clips needs_review → approve 30, reject 20    │
  │                                                                   │
  │  Ngày 2-3: Lặp lại quy trình                                    │
  │  Ngày 4: Kiểm tra quota → ước tính 800-1000 clips happy/surprise │
  │  Ngày 5-7: Bổ sung thiếu nếu cần                                │
  │                                                                   │
  │  Kết quả: ~600 happy + ~450 surprise                             │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 7. Ước Tính Thời Gian & Năng Suất

### 7.1. Thời Gian Xử Lý Của EDS Tool (Cho 1 Video ~30 phút)

| Giai đoạn | Thời gian | Ghi chú |
|:---|:---|:---|
| Tải video YouTube (720p) | 2–5 phút | Tùy tốc độ mạng |
| Phát hiện chuyển cảnh | 30–60 giây | PySceneDetect |
| Smart segmentation (face + dialogue) | 1–3 phút | Quét khuôn mặt @2fps |
| Cắt clips bằng FFmpeg | 10–30 giây | Song song nếu nhiều clips |
| Face extraction per clip | 5–10 giây | MTCNN trên 24 frames |
| Audio extraction per clip | 2–5 giây | FFmpeg + Librosa |
| Whisper transcription per clip | 10–30 giây | GPU: nhanh hơn 5-10x |
| Emotion analysis per clip | 5–15 giây | DeepFace + Lexicon + Wav2Vec2 |
| **TỔNG cho 1 video (20 clips)** | **15–30 phút** | GPU giúp nhanh hơn 2-3x |

### 7.2. Năng Suất Ước Tính

| Chế độ | Videos/đêm | Clips hợp lệ/đêm | Thời gian duyệt sáng | Clips/tuần |
|:---|:---|:---|:---|:---|
| **Thủ công (hiện tại)** | 3–5 | 60–100 | 1–2 giờ | 300–500 |
| **Batch import (nâng cấp)** | 15–25 | 300–500 | 30 phút | 1,500–2,500 |
| **Playlist crawler (nâng cấp)** | 30–50 | 500–800 | 30 phút | 2,500–4,000 |

**Kết luận:** Với nâng cấp Batch Harvest, có thể đạt mục tiêu **3,000 clips** chỉ trong **1–2 tuần** thay vì 4–6 tuần làm thủ công.

### 7.3. Tổng Thời Gian Dự Kiến Toàn Bộ Quy Trình

```
  Timeline dự kiến (với Batch Harvest):
  
  Tuần 1: Thu thập Happy + Surprise       → ~1,050 clips
  Tuần 2: Thu thập Sad + Angry             → ~900 clips
  Tuần 3: Thu thập Fear + Disgust + Neutral → ~1,050 clips
  Tuần 4: Bổ sung thiếu + Kiểm tra chất lượng + Xuất .pkl
  
  → Tổng: 3,000 clips tiếng Việt đã gán nhãn trong 4 tuần
  → Thời gian thao tác thực tế mỗi ngày: ~45 phút
     (15 phút tìm URL tối + 30 phút duyệt sáng)
```

---

## 8. Kiểm Soát Chất Lượng & Lọc Dữ Liệu Xấu

### 8.1. Các Loại Dữ Liệu Xấu Cần Lọc

| Loại | Mô tả | Cách phát hiện tự động | Xử lý |
|:---|:---|:---|:---|
| **No-face clip** | Clip cảnh quan, text overlay, không có mặt người | `num_faces == 0` | Auto-reject |
| **Multi-speaker** | Nhiều người nói đồng thời, cãi nhau | `track_count > 2` | Needs review |
| **Music clip** | Clip ca nhạc, nhạc nền lấn át lời thoại | `has_speech_energy == False` | Auto-reject |
| **Too short** | Clip < 2 giây | `duration < 2.0` | Auto-reject |
| **No speech** | Clip không có lời thoại | `transcript == ""` hoặc `word_count < 2` | Auto-reject |
| **Low confidence** | AI không chắc chắn về cảm xúc | `confidence < 0.40` | Needs review |
| **Incongruent** | Mặt vui nhưng lời buồn (hoặc ngược lại) | `has_incongruity == True` | Needs review |
| **Noisy audio** | Âm thanh nhiễu, không rõ | `audio_clarity < 0.002` | Needs review |

### 8.2. Luồng Lọc Tự Động 3 Tầng

```
  Clip từ Pipeline
       │
       ▼
  ┌─────────────────┐
  │ Tầng 1: Hard    │  → Auto-REJECT nếu:
  │ Reject          │     no face, no speech, too short, no audio
  │ (Loại bỏ ngay)  │
  └────────┬────────┘
           │ (còn lại)
           ▼
  ┌─────────────────┐
  │ Tầng 2: Smart   │  → Auto-APPROVE nếu:
  │ Approve         │     quality ≥ 0.75, confidence ≥ 0.65,
  │ (Duyệt tự động)│     agreement đa số, không incongruity
  └────────┬────────┘
           │ (còn lại)
           ▼
  ┌─────────────────┐
  │ Tầng 3: Human   │  → Hiển thị trên giao diện duyệt
  │ Review          │     Người dùng xem video, chọn nhãn đúng
  │ (Duyệt thủ công)│     Ước tính: ~30-40% clips còn lại
  └─────────────────┘
```

---

## 9. Xử Lý Rào Cản Kỹ Thuật

### 9.1. YouTube Rate Limiting (Giới Hạn Tần Suất)

**Vấn đề:** YouTube có thể chặn IP nếu tải quá nhiều video trong thời gian ngắn (HTTP 429).

**Giải pháp:**
- **Giãn cách tải:** Thêm delay 30-60 giây giữa các lần tải
- **Cookie authentication:** Đăng nhập YouTube → export cookies.txt → cấu hình trong EDS Settings. Tài khoản đã đăng nhập ít bị chặn hơn
- **Tải vào ban đêm:** YouTube ít giám sát vào giờ thấp điểm (0h–6h sáng Việt Nam)
- **Dùng profile "safe":** EDS đã có sẵn profile tải an toàn với retries cao, socket timeout dài

```python
# Cấu hình khuyến nghị cho batch harvest:
# File: tools/emotion-data-studio/.env

DOWNLOAD_MODE=safe              # Profile tải ổn định nhất
DOWNLOAD_MAX_HEIGHT=720         # 720p đủ cho training, tải nhanh hơn 1080p
DOWNLOAD_COOKIES_BROWSER=chrome # Dùng cookies Chrome để tránh bị chặn
```

### 9.2. Dung Lượng Đĩa Cứng

**Ước tính:**
| Thành phần | Dung lượng / 1 video 30p | Dung lượng / 200 videos |
|:---|:---|:---|
| Video gốc (720p) | ~300 MB | ~60 GB |
| Clips cắt nhỏ | ~100 MB | ~20 GB |
| Audio .wav | ~30 MB | ~6 GB |
| Ảnh khuôn mặt | ~20 MB | ~4 GB |
| **Tổng** | **~450 MB** | **~90 GB** |

**Giải pháp:**
- Xóa video gốc sau khi cắt xong clips (tiết kiệm ~60 GB)
- Chỉ giữ clips đã approved, xóa clips rejected
- Sử dụng ổ cứng ngoài nếu cần

### 9.3. Thời Gian GPU vs CPU

| Thành phần | CPU (i7-12700) | GPU (RTX 3060) | Tăng tốc |
|:---|:---|:---|:---|
| Whisper transcription | ~30s / clip | ~5s / clip | **6x** |
| DeepFace analysis | ~8s / clip | ~2s / clip | **4x** |
| MTCNN face detect | ~5s / clip | ~1s / clip | **5x** |
| **Tổng 1 video (20 clips)** | **~25 phút** | **~8 phút** | **~3x** |
| **20 videos/đêm** | ~8 giờ | ~2.5 giờ | **~3x** |

**Khuyến nghị:** Nếu có GPU NVIDIA (bất kỳ), hãy cài PyTorch CUDA để tăng tốc pipeline gấp 3 lần. Với GPU, có thể xử lý **50+ videos/đêm** thay vì 15-20.

### 9.4. Bản Quyền & Đạo Đức

- Dữ liệu chỉ sử dụng cho **mục đích nghiên cứu học thuật** (non-commercial research)
- Không phát hành lại video gốc, chỉ lưu trữ đặc trưng đã trích xuất (features)
- Trong báo cáo, ghi rõ nguồn dữ liệu và mục đích sử dụng
- Tham khảo cách CMU-MOSEI công bố: họ cũng cào video YouTube nhưng chỉ phát hành đặc trưng (file `.pkl`), không phát hành lại video gốc

---

## Tổng Kết

### Chiến Lược Tối Ưu Nhất

```
  1. NGUỒN DỮ LIỆU:
     → Ưu tiên: Phim Việt Nam + Vlog/Reaction + Gameshow trên YouTube
     → Thu thập có mục tiêu theo cảm xúc (nhắm từ khóa tìm kiếm)
     
  2. TỰ ĐỘNG HÓA:
     → Nâng cấp EDS: Batch URL Import + Auto-Queue + Playlist Crawler
     → Smart Auto-Approve 3 tầng (Hard Reject → Smart Approve → Human Review)
     → Dashboard Quota Tracking theo cảm xúc
     
  3. QUY TRÌNH HÀNG NGÀY:
     → Tối: 15 phút tìm URL → batch import → máy chạy qua đêm
     → Sáng: 30 phút duyệt needs_review
     → Tổng: ~45 phút thao tác/ngày → ~300-500 clips hợp lệ/ngày
     
  4. TIMELINE:
     → 4 tuần → 3,000+ clips tiếng Việt đã gán nhãn
     → Đủ để Fine-tune mô hình (Phase 2 trong TRAINING_ROADMAP)
```

### Các Tài Liệu Liên Quan

- [TRAINING_ROADMAP.md](file:///d:/Hai/study/DeepLerning/BCDA/docs/research/TRAINING_ROADMAP.md) — Lộ trình huấn luyện 2 giai đoạn & điều chỉnh EDS.
- [DATASET_PREPARATION.md](file:///d:/Hai/study/DeepLerning/BCDA/docs/research/DATASET_PREPARATION.md) — Chẩn đoán cấu trúc dữ liệu CMU-MOSEI.
- [FINE_TUNING_STRATEGY.md](file:///d:/Hai/study/DeepLerning/BCDA/docs/research/FINE_TUNING_STRATEGY.md) — Kiến trúc mô hình & Transfer Learning.
