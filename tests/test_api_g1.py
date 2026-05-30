import sys
from pathlib import Path
import os
import json

# Thêm project root vào PYTHONPATH để import được backend
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Sử dụng thư mục tạm thời riêng để test để không ảnh hưởng đến dữ liệu thật
os.environ["EDS_DATA_DIR"] = str(Path(project_root) / "data_test")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.main import app
from backend.database.local_db import Base, get_db
from backend.config import settings

# Khởi tạo TestClient
client = TestClient(app)

def test_flow():
    print("=== BẮT ĐẦU CHẠY THỬ NGHIỆM TỰ ĐỘNG GIAI ĐOẠN 1 ===")
    
    # Dọn dẹp database cũ trước khi bắt đầu test
    from backend.database.local_db import engine
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    # 1. Kiểm tra Health Check Endpoint
    print("\n1. Kiểm tra Health Check...")
    response = client.get("/health")
    assert response.status_code == 200, "Health check failed"
    data = response.json()
    print(f"Health check status: {data['status']}")
    print(f"Database status: {data['database']}")
    assert data["status"] == "healthy"
    assert data["database"] == "healthy"
    
    # 2. Kiểm tra Thống kê Dashboard lúc ban đầu (rỗng)
    print("\n2. Kiểm tra Thống kê Dashboard lúc chưa có dữ liệu...")
    response = client.get("/api/labels/stats")
    assert response.status_code == 200
    stats = response.json()
    print(f"Dashboard ban đầu: Total Clips = {stats['total_clips']}, Total Videos = {stats['total_videos']}")
    assert stats["total_clips"] == 0
    assert stats["total_videos"] == 0
    assert len(stats["emotion_distribution"]) == 0
    
    # 3. Tạo Video Mới (YouTube URL)
    print("\n3. Thử nghiệm thêm Video mới...")
    video_payload = {
        "title": "Về Nhà Đi Con - Tập 1",
        "movie_name": "Về Nhà Đi Con",
        "source_url": "https://www.youtube.com/watch?v=mockurl01"
    }
    response = client.post("/api/videos/", json=video_payload)
    assert response.status_code == 200
    video_data = response.json()
    print(f"Thêm video thành công. ID: {video_data['id']}, Trạng thái: {video_data['status']}")
    assert video_data["title"] == "Về Nhà Đi Con - Tập 1"
    assert video_data["status"] == "pending"
    video_id = video_data["id"]
    
    # 4. Kiểm tra danh sách Videos
    print("\n4. Lấy danh sách video để xác minh...")
    response = client.get("/api/videos/")
    assert response.status_code == 200
    videos_list = response.json()
    print(f"Số lượng video trong danh sách: {len(videos_list)}")
    assert len(videos_list) == 1
    assert videos_list[0]["id"] == video_id
    
    # 5. Kích hoạt xử lý Video (Mock)
    print("\n5. Thử kích hoạt xử lý video ngầm (Mock GĐ1)...")
    response = client.post(f"/api/videos/{video_id}/process")
    assert response.status_code == 200
    proc_data = response.json()
    print(f"Video status khi bắt đầu xử lý: {proc_data['status']}")
    assert proc_data["status"] == "processing"
    
    # Đợi 2.5 giây để tiến trình mock chạy ngầm hoàn tất tạo clips
    print("Đang đợi background task giả lập xử lý video...")
    import time
    time.sleep(3.0)
    
    # 6. Kiểm tra lại trạng thái Video sau xử lý
    print("\n6. Kiểm tra trạng thái video sau khi xử lý ngầm hoàn tất...")
    response = client.get(f"/api/videos/{video_id}")
    assert response.status_code == 200
    updated_video = response.json()
    print(f"Trạng thái video mới: {updated_video['status']}, Tổng số clips tạo ra: {updated_video['total_clips']}")
    assert updated_video["status"] == "completed"
    assert updated_video["total_clips"] == 4
    
    # 7. Lấy danh sách clips phân cảnh đã tạo ra
    print("\n7. Lấy danh sách các clips phân cảnh được sinh ra từ video...")
    response = client.get(f"/api/clips/?video_id={video_id}")
    assert response.status_code == 200
    clips_list = response.json()
    print(f"Đã tải {len(clips_list)} clips thành công:")
    for clip in clips_list:
        print(f"  - Clip #{clip['clip_index']}: Emotion = {clip['predicted_emotion']}, Conf = {clip['confidence']}, Text = '{clip['transcript']}'")
    assert len(clips_list) == 4
    clip_id_to_review = clips_list[0]["id"]
    
    # 8. Thực hiện kiểm duyệt thủ công (Review) 1 clip
    print("\n8. Thực hiện kiểm duyệt thủ công (Review & Approve) clip đầu tiên...")
    review_payload = {
        "status": "approved",
        "user_emotion": "sad",
        "reviewer_notes": "Nhãn do AI gán chuẩn xác, nét mặt buồn bã rất rõ."
    }
    response = client.put(f"/api/clips/{clip_id_to_review}", json=review_payload)
    assert response.status_code == 200
    reviewed_clip = response.json()
    print(f"Duyệt clip thành công. Mới: status={reviewed_clip['status']}, user_emotion={reviewed_clip['user_emotion']}")
    assert reviewed_clip["status"] == "approved"
    assert reviewed_clip["user_emotion"] == "sad"
    
    # 9. Kiểm tra xem approved_clips trên Video có được đồng bộ tự động không
    print("\n9. Kiểm tra đồng bộ số lượng clips đã phê duyệt ở bảng Video...")
    response = client.get(f"/api/videos/{video_id}")
    assert response.status_code == 200
    video_after_review = response.json()
    print(f"Approved clips trên Video: {video_after_review['approved_clips']}")
    assert video_after_review["approved_clips"] == 1
    
    # 10. Thống kê Dashboard sau khi đã duyệt dữ liệu
    print("\n10. Thống kê Dashboard sau khi đã có dữ liệu duyệt...")
    response = client.get("/api/labels/stats")
    assert response.status_code == 200
    stats_after = response.json()
    print(f"Thống kê Dashboard mới: Total Clips = {stats_after['total_clips']}, Approved = {stats_after['approved_clips']}")
    assert stats_after["total_clips"] == 4
    assert stats_after["approved_clips"] == 1
    print("Phân bố cảm xúc nhận được:")
    for dist in stats_after["emotion_distribution"]:
        print(f"  - Cảm xúc '{dist['emotion']}': {dist['count']} clips ({dist['percentage']}%)")
    assert len(stats_after["emotion_distribution"]) > 0
    
    # 11. Xóa Video
    print("\n11. Thử xóa video gốc để xác minh cascade delete...")
    response = client.delete(f"/api/videos/{video_id}")
    assert response.status_code == 200
    print("Đã xóa video thành công.")
    
    # Xác minh danh sách clips đã rỗng
    response = client.get(f"/api/clips/?video_id={video_id}")
    assert response.status_code == 200
    assert len(response.json()) == 0
    print("Xác nhận toàn bộ clips đi kèm đã được cascade delete sạch sẽ!")
    
    print("\n=== HOÀN TẤT TẤT CẢ CÁC BÀI KIỂM TRA: THÀNH CÔNG RỰC RỠ! ===")

if __name__ == "__main__":
    test_flow()
    
    # Giải phóng connection pool của SQLAlchemy để đóng kết nối SQLite
    from backend.database.local_db import engine
    engine.dispose()
    
    # Dọn dẹp thư mục test sau khi test xong
    import shutil
    test_dir = Path(project_root) / "data_test"
    if test_dir.exists():
        try:
            shutil.rmtree(test_dir)
            print("Đã dọn dẹp thư mục dữ liệu test tạm thời.")
        except Exception as e:
            print(f"Cảnh báo: Không thể dọn dẹp thư mục test: {e}")
