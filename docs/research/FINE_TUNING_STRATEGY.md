# 🎯 Chiến Lược Kiến Trúc Mô Hình & Học Chuyển Giao (Fine-tuning)
## Đề Tài 17: Hệ Thống Phân Tích Cảm Xúc Đa Phương Thức (Multimodal Emotion Analysis)

Tài liệu này tập trung thiết kế ý tưởng kiến trúc mô hình học sâu đa phương thức (kết hợp Hình ảnh, Âm thanh, Văn bản) và phương pháp áp dụng kỹ thuật **Học chuyển giao (Transfer Learning)** kết hợp **Tinh chỉnh (Fine-tuning)** để tối ưu hóa độ chính xác nhận diện cảm xúc tiếng Việt.

---

## 1. Kiến Trúc Mô Hình Đa Phương Thức (Multimodal Architecture)

Để giải quyết bài toán phân tích cảm xúc từ Video hội thoại tiếng Việt, mô hình được thiết kế gồm 3 nhánh trích xuất đặc trưng độc lập (Encoders) và 1 mạng kết hợp thông tin (Fusion Network).

```
 ┌───────────────┐
 │   Khung ảnh   ├─► [ Nhánh Video: CNN + LSTM ] ──────┐
 └───────────────┘                                     │
 ┌───────────────┐                                     ▼
 │   Tín hiệu âm ├─► [ Nhánh Audio: CNN/LSTM/Wav2Vec ] ─┼─► [ Mạng Kết Hợp ] ─► [ Phân loại Cảm xúc ]
 └───────────────┘                                     ▲   (Fusion Network)    (7 nhãn cảm xúc)
 ┌───────────────┐                                     │
 │ Văn bản tiếng ├─► [ Nhánh Text: PhoBERT / LSTM ] ───┘
 └───────────────┘
```

### 1.1. Nhánh Xử Lý Hình Ảnh (Video Branch)
* **Mục tiêu:** Trích xuất sự thay đổi biểu cảm khuôn mặt qua thời gian từ chuỗi khung hình (frame).
* **Kiến trúc:** **CNN (Spatial Encoder) + LSTM (Temporal Encoder)**.
  * **CNN (2D):** Sử dụng các mô hình pre-trained như **ResNet-50**, **MobileNetV3** hoặc **FaceNet** để chuyển mỗi khuôn mặt được trích xuất thành một vector đặc trưng không gian.
  * **LSTM/GRU (1D):** Nhận chuỗi các vector đặc trưng từ CNN theo trình tự thời gian để mô hình hóa sự thay đổi biểu cảm (nhíu mày, cười, mở to mắt) và xuất ra một vector biểu diễn chuyển động khuôn mặt.

### 1.2. Nhánh Xử Lý Âm Thanh (Audio Branch)
* **Mục tiêu:** Trích xuất ngữ điệu, tần số và cường độ giọng nói của người nói.
* **Kiến trúc:** **MFCC + LSTM** hoặc **Wav2Vec2.0**.
  * **Phương pháp truyền thống:** Trích xuất đặc trưng **MFCC (Mel-Frequency Cepstral Coefficients)** từ file âm thanh `.wav`, đưa qua mạng **LSTM** hoặc mạng chập **1D CNN** để học sự biến thiên của giọng nói.
  * **Phương pháp học chuyển giao:** Sử dụng mô hình pre-trained như `wav2vec2-large-vietnamese` để trích xuất trực tiếp đặc trưng âm sắc tiếng Việt chất lượng cao.

### 1.3. Nhánh Xử Lý Văn Bản (Text Branch)
* **Mục tiêu:** Hiểu ngữ nghĩa của câu hội thoại bằng tiếng Việt.
* **Kiến trúc:** **Word Embedding + LSTM** hoặc **Transformer (PhoBERT)**.
  * **Phương pháp truyền thống:** Sử dụng Word2Vec/FastText tiếng Việt làm đầu vào, đi qua mạng **Bi-LSTM** (LSTM hai chiều) để nắm bắt ngữ cảnh từ cả hai phía.
  * **Phương pháp học chuyển giao:** Sử dụng mô hình `vinai/phobert-base` (được huấn luyện tối ưu riêng cho tiếng Việt) để chuyển đổi câu thoại thành vector ngữ nghĩa cô đọng.

### 1.4. Mạng Kết Hợp Đặc Trưng (Fusion Network)
Sau khi có 3 vector đặc trưng tương ứng từ 3 nhánh (Video, Audio, Text), ta có các chiến lược kết hợp thông tin sau:
1. **Early Fusion (Feature Concatenation):** Nối (concatenate) trực tiếp 3 vector đặc trưng lại với nhau thành một vector dài duy nhất, sau đó đưa qua các lớp Fully Connected (Tuyến tính) để phân loại.
2. **Late Fusion (Decision/Voting):** Cho mỗi nhánh tự đưa ra dự đoán cảm xúc (xác suất của 7 cảm xúc), sau đó lấy trung bình cộng có trọng số hoặc dùng cơ chế bỏ phiếu (Ensemble Voting) để đưa ra kết quả cuối cùng.
3. **Attention-based Fusion:** Sử dụng cơ chế Cross-Attention để mô hình tự học xem tại thời điểm nào thì biểu cảm mặt (Video), tông giọng (Audio) hay từ ngữ (Text) là quan trọng nhất để quyết định cảm xúc.

---

## 2. Chiến Lược Học Chuyển Giao & Tinh Chỉnh (Fine-tuning)

Do tập dữ liệu tiếng Việt tự cào thường có quy mô nhỏ (vài trăm mẫu), việc huấn luyện tất cả các tham số của cả 3 nhánh từ đầu sẽ dẫn đến **Overfitting** (mô hình học vẹt dữ liệu cũ, không đoán được dữ liệu mới). Chiến lược Fine-tuning giải quyết điều này bằng cách chia mô hình làm 2 phần:

### 2.1. Đóng băng Bộ Trích Xuất Đặc Trưng (Feature Extractor Freezing)
* **Nhánh Video:** Đóng băng toàn bộ trọng số của mô hình CNN (ví dụ ResNet50). Các bộ lọc tìm góc cảnh, nếp nhăn, mắt, mũi, miệng vốn đã rất hoàn hảo trên dữ liệu quốc tế sẽ được giữ nguyên, không thay đổi.
* **Nhánh Text:** Đóng băng hầu hết các tầng của PhoBERT, chỉ mở khóa (unfreeze) 1 hoặc 2 tầng Transformer cuối cùng.
* **Nhánh Audio:** Đóng băng phần trích xuất Wav2Vec.

### 2.2. Huấn luyện tầng Kết hợp & Phân loại (Fusion Layer & Classifier Fine-tuning)
* Huấn luyện các trọng số của lớp **Fusion Network** và lớp **Fully Connected** cuối cùng.
* Sử dụng tốc độ học (Learning Rate) rất nhỏ (ví dụ: `lr = 1e-4` hoặc `1e-5`) để mô hình cập nhật nhẹ nhàng mà không phá vỡ các đặc trưng cơ bản đã học được từ trước.

---

## 3. Giải Thích Khoa Học & Phân Tích Tính Khả Thi (Báo Cáo & Thuyết Trình)

> [!NOTE]
> *Phần này cung cấp các cơ sở lý thuyết khoa học và lập luận thực tế để bạn đưa vào báo cáo và trả lời phản biện trước hội đồng giám khảo.*

### 3.1. Tại sao chuyển đổi từ dữ liệu nước ngoài sang người Việt lại khả thi?

#### A. Tính Phổ Quát Toàn Cầu của Biểu Cảm Khuôn Mặt (Universal Facial Expressions)
Theo nghiên cứu kinh điển của nhà tâm lý học **Paul Ekman**, con người bất kể chủng tộc (châu Âu, châu Á, châu Phi) đều biểu lộ các cảm xúc cơ bản thông qua các nhóm cơ mặt giống nhau (gọi là **Action Units**):
* Khi **Giận dữ**, cơ chân mày co lại (nhíu mày) và môi mím chặt.
* Khi **Vui vẻ**, cơ gò má nâng lên và khóe miệng kéo lên (tạo nụ cười).
* Khi **Ngạc nhiên**, mắt mở to và miệng trễ xuống.

Các mô hình AI nhận diện biểu cảm (như CNN) thực chất học các **chuyển động hình học** này chứ không học màu da hay cấu trúc xương chủng tộc. 
* *Sự khác biệt về chủng tộc nằm ở cấu trúc hình học cố định (khoảng cách mắt, mũi, cằm) - cái này chỉ ảnh hưởng đến bài toán **Nhận dạng danh tính (Face Recognition)**.*
* *Đối với bài toán **Nhận diện cảm xúc (Emotion Recognition)**, mô hình học từ khuôn mặt người phương Tây vẫn hoàn toàn nhận biết chính xác cảm xúc người Việt.*

#### B. Sự Khác Biệt Nằm ở Đâu và Giải Quyết Thế Nào?
Sự khác biệt thực sự nằm ở 2 yếu tố:
1. **Văn hóa biểu cảm:** Người châu Á/người Việt thường biểu cảm vi tế, kín đáo hơn (micro-expressions), trong khi người phương Tây thường biểu cảm phóng đại và rõ nét hơn.
2. **Giọng nói (Audio):** Tiếng Việt là ngôn ngữ đơn âm tiết và có thanh điệu (sắc, huyền, hỏi, ngã, nặng, ngang), khác hoàn toàn với ngữ điệu đa âm tiết của tiếng Anh.

**Giải pháp:** Bằng cách **dịch phần Text** (giữ nguyên hình ảnh/âm thanh của dataset thế giới) để làm Pre-training, mô hình học được mối liên kết cơ bản giữa (Từ ngữ dịch + Biểu cảm chuẩn). Sau đó, dùng dữ liệu cào nhỏ của người Việt để **Fine-tune** lại lớp cuối, mô hình sẽ tự căn chỉnh theo văn hóa và ngữ điệu nói của người Việt.

---

### 3.2. Bản chất kỹ thuật của Transfer Learning qua Ẩn dụ trực quan

#### Ẩn dụ "Đầu bếp và Bánh mì Việt Nam"
Để có một đầu bếp chuyên làm bánh mì Việt Nam chất lượng:
* **Nếu huấn luyện từ đầu (From Scratch):** Bạn nhận một đứa trẻ chưa biết nấu ăn, dạy cách dùng dao, cách đo nhiệt độ lò, cách nhào bột, phân biệt gia vị... việc này mất **3 đến 5 năm** và tốn rất nhiều nguyên liệu thử nghiệm thất bại.
* **Nếu áp dụng Fine-tuning:** Bạn tuyển một **đầu bếp phương Tây đã lành nghề** (họ đã cực giỏi kỹ năng cầm dao, nhào bột và chỉnh nhiệt độ lò nướng từ trước). Bạn chỉ cần dạy họ công thức làm nhân pate, nước sốt kiểu Việt Nam trong **3 ngày**. Họ học cực kỳ nhanh và tốn rất ít nguyên liệu.

#### Áp dụng vào Mạng Nơ-ron (Neural Network):
Mô hình AI huấn luyện trên hàng triệu khuôn mặt thế giới giống như người đầu bếp phương Tây lành nghề kia. 
* Các **tầng mạng nông và trung gian** đóng vai trò là "kỹ năng cơ bản" (phát hiện đường nét, mắt, mũi, miệng). Chúng ta **đóng băng (freeze)** phần này.
* Tầng **Fully Connected cuối cùng** đóng vai trò là "công thức nước sốt Việt Nam". Chúng ta chỉ huấn luyện lại phần này bằng dữ liệu tiếng Việt. Điều này giúp mô hình đạt độ chính xác cực cao chỉ với một tập dữ liệu rất nhỏ.

---

## 4. Quy Trình Huấn Luyện Đề Xuất (Roadmap)

1. **Bước 1 (Pre-training):**
   * Huấn luyện toàn bộ mạng Fusion trên các tập dữ liệu đa phương thức chuẩn thế giới đã được dịch tự động phần Text sang tiếng Việt (như **MELD** hoặc **CMU-MOSEI**).
   * Mục tiêu: Giúp mạng Fusion học cách kết hợp thông tin đa phương thức hiệu quả.
2. **Bước 2 (Domain Adaptation / Fine-tuning):**
   * Đóng băng các nhánh chính (Backbones).
   * Huấn luyện mô hình trên tập dữ liệu người Việt thực tế (cào từ các talkshow, video phỏng vấn tiếng Việt) để học các biểu cảm tinh tế và ngữ điệu nói đặc thù của người Việt.

---

## 5. Thiết Kế Mã Nguồn Minh Họa (PyTorch Implementation)

Dưới đây là thiết kế kiến trúc mô hình kết hợp (Early Fusion) bằng PyTorch hỗ trợ cơ chế Đóng băng và Tinh chỉnh:

```python
import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel

class MultimodalEmotionClassifier(nn.Module):
    def __init__(self, num_classes=7, freeze_backbones=True):
        super(MultimodalEmotionClassifier, self).__init__()
        
        # === 1. NHÁNH VIDEO BACKBONE (ResNet50) ===
        # Trích xuất đặc trưng hình ảnh từ khuôn mặt
        resnet = models.resnet50(pretrained=True)
        self.video_backbone = nn.Sequential(*list(resnet.children())[:-1]) # Bỏ lớp linear cuối
        self.video_feature_dim = 2048
        
        # === 2. NHÁNH TEXT BACKBONE (PhoBERT) ===
        # Trích xuất đặc trưng ngữ nghĩa tiếng Việt
        self.text_backbone = AutoModel.from_pretrained("vinai/phobert-base")
        self.text_feature_dim = 768
        
        # === 3. NHÁNH AUDIO BACKBONE (CNN 1D cho MFCC) ===
        # Giả định đầu vào MFCC có kích thước (batch, 40, time_steps)
        self.audio_extractor = nn.Sequential(
            nn.Conv1d(40, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Rút gọn chiều thời gian về 1
        )
        self.audio_feature_dim = 64
        
        # === 4. ĐÓNG BĂNG TRỌNG SỐ BACKBONES ===
        if freeze_backbones:
            for param in self.video_backbone.parameters():
                param.requires_grad = False
            for param in self.text_backbone.parameters():
                param.requires_grad = False
                
        # === 5. MẠNG KẾT HỢP (FUSION & CLASSIFIER) ===
        combined_dim = self.video_feature_dim + self.text_feature_dim + self.audio_feature_dim
        
        self.fusion_network = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes) # Output 7 nhãn cảm xúc
        )

    def forward(self, face_image, text_input_ids, text_attention_mask, audio_mfcc):
        # 1. Trích xuất đặc trưng Video (Visual)
        with torch.no_grad(): # Tối ưu bộ nhớ vì đã đóng băng
            video_feats = self.video_backbone(face_image) # Shape: (batch, 2048, 1, 1)
            video_feats = torch.flatten(video_feats, 1) # Shape: (batch, 2048)
            
        # 2. Trích xuất đặc trưng Text (Ngữ nghĩa)
        with torch.no_grad():
            text_outputs = self.text_backbone(input_ids=text_input_ids, attention_mask=text_attention_mask)
            # Lấy đặc trưng của token [CLS] đại diện cho cả câu
            text_feats = text_outputs.last_hidden_state[:, 0, :] # Shape: (batch, 768)
            
        # 3. Trích xuất đặc trưng Audio
        audio_feats = self.audio_extractor(audio_mfcc) # Shape: (batch, 64, 1)
        audio_feats = torch.flatten(audio_feats, 1) # Shape: (batch, 64)
        
        # 4. Kết hợp đặc trưng (Early Fusion)
        combined_feats = torch.cat((video_feats, text_feats, audio_feats), dim=1) # Shape: (batch, 2048 + 768 + 64)
        
        # 5. Phân loại cảm xúc
        output_logits = self.fusion_network(combined_feats)
        return output_logits

# === HUẤN LUYỆN TINH CHỈNH ===
# Chỉ truyền các tham số của các tầng KHÔNG bị đóng băng vào Optimizer
model = MultimodalEmotionClassifier(num_classes=7, freeze_backbones=True)
trainable_parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_parameters, lr=1e-4)
```
