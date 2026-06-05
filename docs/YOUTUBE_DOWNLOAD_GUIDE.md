# Kế Hoạch Nâng Cấp Chức Năng Tải Video Từ URL Cho `emotion-data-studio` (2026)

Tài liệu này chuyển từ dạng nghiên cứu chung sang **kế hoạch triển khai thực tế cho codebase hiện tại** của `tools/emotion-data-studio`.

Mục tiêu là nâng cấp chức năng tải video từ URL để:

- tải **ổn định hơn** với YouTube và các URL do `yt-dlp` hỗ trợ
- tăng khả năng vượt các lỗi phổ biến như `403`, `429`, `signature extraction failed`, throttling
- tận dụng tốt hơn tài nguyên máy (`FFmpeg`, kết nối phân đoạn, downloader ngoài)
- hỗ trợ rõ ràng các tình huống cần `cookies`
- có **fallback nhiều tầng**, log dễ chẩn đoán, phù hợp với UI desktop hiện tại

---

## 1. Phạm vi triển khai trong codebase

Các file liên quan trực tiếp:

- `tools/emotion-data-studio/backend/services/downloader.py`
- `tools/emotion-data-studio/backend/services/pipeline_orchestrator.py`
- `tools/emotion-data-studio/backend/config.py`
- `tools/emotion-data-studio/ui/pages/settings_page.py`
- `tools/emotion-data-studio/docs/SETUP_GUIDE.md`
- tài liệu này: `docs/YOUTUBE_DOWNLOAD_GUIDE.md`

---

## 2. Thực trạng hiện tại

### 2.1. Điểm đã có

`VideoDownloader` hiện đã làm được:

- xác thực URL cơ bản
- lấy metadata bằng `yt-dlp`
- tải video tối đa `720p`
- hỗ trợ merge với `FFmpeg`
- có `concurrent_fragment_downloads`, `buffersize`, `socket_timeout`
- có `progress_hook` để cập nhật UI

### 2.2. Hạn chế hiện tại

Phần tải URL vẫn còn đơn giản ở các điểm sau:

1. **Chưa có extractor args cho YouTube**
   - chưa ưu tiên `player_client`
   - chưa có chiến lược hạn chế lỗi chữ ký/PO token

2. **Chưa có retry strategy đủ mạnh**
   - chưa phân biệt lỗi mạng tạm thời, lỗi bị chặn, lỗi định dạng
   - chưa có fallback cấu hình theo từng lần thử

3. **Chưa hỗ trợ cookie/browser session theo cấu hình người dùng**
   - video giới hạn tuổi, riêng tư, hoặc bị YouTube chặn sẽ dễ fail

4. **Chưa có downloader ngoài tùy chọn**
   - chưa hỗ trợ `aria2c` để tăng tốc các trường hợp mạng mạnh

5. **Chưa có chế độ “tải an toàn” và “tải tăng tốc”**
   - hiện mọi URL dùng gần như một cấu hình duy nhất

6. **Chưa có logging chẩn đoán đủ rõ**
   - người dùng khó biết lỗi do `FFmpeg`, `cookies`, `yt-dlp`, `Deno`, `403`, `429`, hay throttling

---

## 3. Mục tiêu kỹ thuật sau nâng cấp

Sau khi triển khai, chức năng tải URL nên đạt:

### Mức 1 — Ổn định mặc định

- tải tốt đa số video công khai
- fallback tự động nếu cấu hình đầu tiên thất bại
- báo lỗi rõ ràng bằng tiếng Việt

### Mức 2 — Chống throttling tốt hơn

- tăng tốc tải DASH bằng tải phân đoạn song song
- tùy chọn dùng `aria2c`
- cấu hình `throttled_rate` để reconnect

### Mức 3 — Giảm lỗi 403/429 thực tế

- hỗ trợ `player_client`
- hỗ trợ `cookies-from-browser`
- hỗ trợ `cookiefile`
- khuyến nghị cài `Deno`

### Mức 4 — Tích hợp UX tốt hơn cho desktop tool

- chọn chế độ tải trong `Settings`
- lưu cấu hình downloader vào `user_settings.json`
- hiển thị lỗi và gợi ý khắc phục ngay trên UI

---

## 4. Kiến trúc đề xuất cho downloader mới

### 4.1. Tách downloader thành nhiều profile chiến lược

Thay vì một cấu hình `ydl_opts` cố định, nên có các profile:

#### `balanced`
Dùng mặc định cho đa số người dùng:

- 720p
- concurrent fragment downloads vừa phải
- merge mp4
- retry cơ bản
- extractor args an toàn

#### `safe`
Ưu tiên ổn định hơn tốc độ:

- giảm số luồng tải phân đoạn
- tăng retry
- timeout dài hơn
- ưu tiên format merge sẵn nếu cần

#### `turbo`
Ưu tiên tốc độ trên máy tốt / mạng tốt:

- tăng `concurrent_fragment_downloads`
- dùng `aria2c` nếu khả dụng
- tăng `buffersize`

#### `cookies_required`
Dùng khi video bị giới hạn:

- nạp cookie từ browser hoặc cookie file
- ưu tiên `web_embedded`, `android`

---

### 4.2. Retry theo tầng

Đề xuất flow tải như sau:

1. **Attempt 1**: `balanced`
2. **Attempt 2**: `balanced + player_client tuned`
3. **Attempt 3**: `safe`
4. **Attempt 4**: `cookies` nếu người dùng đã cấu hình

Ví dụ logic:

```python
profiles = [
    self._build_profile("balanced"),
    self._build_profile("balanced_fallback"),
    self._build_profile("safe"),
]

if cookie_available:
    profiles.append(self._build_profile("cookies"))
```

Mỗi lần fail cần ghi:

- profile đã dùng
- lỗi rút gọn
- có nên thử lần sau không

---

## 5. Cấu hình `yt-dlp` khuyến nghị cho project này

### 5.1. Cấu hình lõi

```python
ydl_opts = {
    "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
    "outtmpl": output_template,
    "merge_output_format": "mp4",
    "quiet": True,
    "no_warnings": True,
    "nocheckcertificate": True,
    "concurrent_fragment_downloads": 5,
    "buffersize": 262144,
    "socket_timeout": 20,
    "retries": 8,
    "fragment_retries": 8,
    "extractor_retries": 3,
    "file_access_retries": 3,
    "throttled_rate": 102400,
}
```

### 5.2. Extractor args cho YouTube

```python
"extractor_args": {
    "youtube": {
        "player_client": ["android", "web_embedded", "tv"],
        "player_js_version": ["actual"],
    }
}
```

Ghi chú:

- Không phải video nào cũng cần tất cả client.
- Nên để client ưu tiên qua cấu hình, không hard-code cứng một lựa chọn duy nhất.

### 5.3. Downloader ngoài `aria2c`

Nếu có `aria2c`, có thể thêm profile tăng tốc:

```python
"external_downloader": "aria2c",
"external_downloader_args": {
    "aria2c": ["-x", "16", "-s", "16", "-k", "1M"]
}
```

Chỉ nên bật nếu:

- máy đủ mạnh
- mạng ổn định
- `aria2c` đã cài thật

---

## 6. Cookies và session người dùng

### 6.1. Trường hợp cần cookies

Nên hỗ trợ cookies khi gặp:

- video giới hạn độ tuổi
- video chỉ hiển thị khi đăng nhập
- video public nhưng `403` lặp lại với cấu hình thường

### 6.2. Hai cách hỗ trợ trong tool

#### Cách A — Tự lấy từ browser

Cấu hình trong `yt-dlp`:

```python
"cookiesfrombrowser": ("chrome",)
```

Nên cho người dùng chọn trong Settings:

- `Không dùng`
- `Chrome`
- `Edge`
- `Firefox`

#### Cách B — Dùng cookie file

Cấu hình:

```python
"cookiefile": "path/to/youtube_cookies.txt"
```

Nên ưu tiên khi:

- app chạy portable
- app chạy trên máy không có browser session ổn định
- người dùng muốn kiểm soát tài khoản cụ thể

### 6.3. Lưu ý bảo mật

- không copy cookie vào repo
- không log raw cookie
- chỉ lưu đường dẫn file cookie, không lưu nội dung
- cần warning trên UI rằng cookie có thể chứa session đăng nhập

---

## 7. Deno và môi trường runtime phụ trợ

### 7.1. Khuyến nghị

Trên Windows, nên khuyến nghị cài:

```powershell
winget install DenoLand.Deno
```

Lý do:

- hỗ trợ tốt hơn cho một số tình huống giải mã player JS / signature
- giảm xác suất lỗi do thay đổi phía YouTube

### 7.2. Không bắt buộc cứng trong code

Tool không nên fail chỉ vì thiếu `Deno`.

Thay vào đó:

- vẫn thử tải bình thường
- nếu fail, hiển thị gợi ý cài `Deno`
- diagnostics trong Settings nên có mục kiểm tra `Deno`

---

## 8. Nâng cấp Settings UI đề xuất

Trong `settings_page.py`, nên thêm nhóm cấu hình mới:

### Nhóm: `Tải Video Từ URL`

Các field đề xuất:

- `Chế độ tải`:
  - `Cân bằng`
  - `An toàn`
  - `Tăng tốc`

- `Độ phân giải tối đa`:
  - `480p`
  - `720p`
  - `1080p` (chỉ khi đã có FFmpeg)

- `Số phân đoạn tải song song`
- `Sử dụng aria2c nếu có`
- `Ngưỡng throttling để tự kết nối lại`
- `Trình duyệt lấy cookies`
- `Đường dẫn file cookies.txt`
- `Cho phép fallback client YouTube`

Các key nên lưu vào `user_settings.json`:

```json
{
  "download_mode": "balanced",
  "download_max_height": 720,
  "download_concurrent_fragments": 5,
  "download_use_aria2": true,
  "download_throttled_rate_kbps": 100,
  "download_cookies_browser": "chrome",
  "download_cookie_file": "",
  "download_enable_youtube_fallback_clients": true
}
```

---

## 9. Nâng cấp `VideoDownloader` đề xuất

### 9.1. Tách hàm build options

Nên refactor `downloader.py` theo cấu trúc:

```python
def _build_base_opts(self) -> dict: ...
def _build_format_spec(self, ffmpeg_available: bool, max_height: int) -> str: ...
def _apply_cookie_strategy(self, opts: dict) -> dict: ...
def _apply_youtube_strategy(self, opts: dict, profile: str) -> dict: ...
def _download_with_profiles(self, url: str, profiles: list[str], progress_hook=None) -> dict: ...
```

### 9.2. Chuẩn hóa lỗi

Nên map lỗi `yt-dlp` sang thông báo tiếng Việt thân thiện:

- `403` -> `YouTube từ chối truy cập. Hãy thử bật cookies hoặc đổi profile tải.`
- `429` -> `YouTube tạm chặn do quá nhiều yêu cầu. Hãy chờ và thử lại.`
- `Sign in to confirm your age` -> `Video yêu cầu đăng nhập hoặc xác minh độ tuổi.`
- `Requested format is not available` -> `Không tìm thấy định dạng video phù hợp.`
- `ffmpeg not found` -> `Thiếu FFmpeg để ghép audio/video chất lượng cao.`

### 9.3. Cache info tốt hơn

Ngoài `_info_cache`, nên cache ngắn hạn:

- `extract_info` result
- thông tin file đã tải xong
- mapping URL -> video_id -> local_path

Mục tiêu là tránh gọi lại YouTube không cần thiết.

---

## 10. Kịch bản fallback triển khai khuyến nghị

### Kịch bản A — Video công khai bình thường

- dùng `balanced`
- `player_client = android, web_embedded`
- merge mp4

### Kịch bản B — Video bị throttling

- tăng `concurrent_fragment_downloads`
- bật `throttled_rate`
- nếu có `aria2c` thì chuyển `turbo`

### Kịch bản C — Video lỗi `403`

- thử client fallback
- thử cookies từ browser nếu đã cấu hình
- thử cookie file nếu có
- gợi ý kiểm tra `Deno`

### Kịch bản D — Không có FFmpeg

- chỉ tải `best` dạng merge sẵn
- warning rằng chất lượng có thể thấp hơn

---

## 11. Logging và chẩn đoán nên có

Khi tải URL, log nên ghi rõ:

- URL domain
- profile hiện dùng
- format spec
- có/không có FFmpeg
- có/không có aria2c
- có/không dùng cookies
- lỗi rút gọn của từng attempt

Ví dụ log tốt:

```text
[Downloader] URL accepted: youtube.com
[Downloader] Profile: balanced
[Downloader] FFmpeg: yes
[Downloader] aria2c: no
[Downloader] Cookies: browser=chrome
[Downloader] Attempt 1 failed: HTTP Error 403
[Downloader] Attempt 2 profile=balanced_fallback
[Downloader] Download success: 1280x720, merged mp4
```

---

## 12. Lộ trình triển khai theo thứ tự

### Giai đoạn 1 — Ổn định tối thiểu

1. Refactor `downloader.py` thành builder + retry profile
2. Thêm `extractor_args.youtube.player_client`
3. Thêm `retries`, `fragment_retries`, `extractor_retries`
4. Chuẩn hóa lỗi tiếng Việt

### Giai đoạn 2 — Tăng tốc

5. Thêm tùy chọn `aria2c`
6. Thêm `throttled_rate`
7. Cho chọn `max_height` và `concurrent fragments`

### Giai đoạn 3 — Session/cookies

8. Hỗ trợ `cookiesfrombrowser`
9. Hỗ trợ `cookiefile`
10. Cảnh báo bảo mật trên UI

### Giai đoạn 4 — Diagnostics & UX

11. Settings UI cho download profile
12. Diagnostics kiểm tra `FFmpeg`, `aria2c`, `Deno`, cookies path
13. Hiển thị gợi ý khắc phục ngay trên màn hình nhập URL

---

## 13. Mẫu cấu hình Python đề xuất cho `emotion-data-studio`

```python
def build_ydl_opts(output_template: str, ffmpeg_available: bool) -> dict:
    opts = {
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "socket_timeout": 20,
        "retries": 8,
        "fragment_retries": 8,
        "extractor_retries": 3,
        "buffersize": 262144,
        "concurrent_fragment_downloads": 5,
        "throttled_rate": 102400,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web_embedded"],
                "player_js_version": ["actual"],
            }
        },
    }

    if ffmpeg_available:
        opts["format"] = "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best"
        opts["merge_output_format"] = "mp4"
    else:
        opts["format"] = "best[height<=720][ext=mp4]/best"

    return opts
```

---

## 14. Kết luận

Đối với `tools/emotion-data-studio`, hướng nâng cấp đúng không chỉ là “tăng tốc tải”, mà là xây dựng một downloader có đủ 4 lớp:

1. **Ổn định mặc định**
2. **Tăng tốc khi môi trường tốt**
3. **Fallback nhiều tầng khi YouTube chặn**
4. **Cấu hình được từ UI và chẩn đoán được rõ ràng**

Nếu triển khai đúng theo tài liệu này, chức năng tải URL của tool sẽ:

- đỡ lỗi hơn với YouTube 2026
- nhanh hơn trên máy có FFmpeg/aria2c
- thân thiện hơn với người dùng desktop
- dễ mở rộng sang các nền tảng khác do `yt-dlp` hỗ trợ

---

## 15. Hành động tiếp theo được khuyến nghị

Thứ tự triển khai tiếp theo nên là:

1. refactor `backend/services/downloader.py`
2. thêm settings download profile vào `settings_page.py`
3. thêm diagnostics cho `FFmpeg`, `aria2c`, `Deno`, cookies
4. cập nhật `SETUP_GUIDE.md` cho phần cài `Deno`, `aria2c`, cookie workflow
5. test thực tế với:
   - 1 video công khai bình thường
   - 1 video age-restricted
   - 1 video bị throttling
   - 1 video không có FFmpeg
