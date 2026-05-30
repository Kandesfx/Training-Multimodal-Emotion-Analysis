"""
Cấu hình trung tâm cho toàn bộ dự án.
Tất cả hyperparameters, đường dẫn, và thiết lập model đều ở đây.

Giải thích:
- Dùng @dataclass để tạo config dạng class, dễ quản lý hơn dict
- Mỗi nhóm config tách riêng để dễ tìm và sửa
- Có thể override bằng command line args khi cần
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os


# ============================================
# Đường dẫn gốc của dự án
# ============================================
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PathConfig:
    """Các đường dẫn thư mục trong dự án."""
    
    # Dữ liệu
    data_root: Path = PROJECT_ROOT / "data"
    raw_video: Path = PROJECT_ROOT / "data" / "raw"
    processed: Path = PROJECT_ROOT / "data" / "processed"
    video_frames: Path = PROJECT_ROOT / "data" / "processed" / "video_frames"
    audio: Path = PROJECT_ROOT / "data" / "processed" / "audio"
    transcripts: Path = PROJECT_ROOT / "data" / "processed" / "transcripts"
    labels: Path = PROJECT_ROOT / "data" / "labels"
    splits: Path = PROJECT_ROOT / "data" / "splits"
    
    # Model
    checkpoints: Path = PROJECT_ROOT / "checkpoints"
    logs: Path = PROJECT_ROOT / "logs"
    
    def create_all(self):
        """Tạo tất cả thư mục nếu chưa tồn tại."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Path):
                field_value.mkdir(parents=True, exist_ok=True)


@dataclass
class VideoModuleConfig:
    """Cấu hình cho module Video (CNN - ResNet50 + Temporal LSTM)."""
    
    # Model
    backbone: str = "resnet50"           # resnet18, resnet34, resnet50
    pretrained: bool = True              # Dùng pretrained ImageNet weights
    feature_dim: int = 512               # Kích thước vector đặc trưng đầu ra
    temporal_hidden: int = 256           # Hidden size cho Temporal LSTM
    temporal_layers: int = 1             # Số layer Temporal LSTM
    dropout: float = 0.3                 # Tỷ lệ dropout để tránh overfitting
    freeze_backbone_layers: int = 6      # Freeze N layers đầu của ResNet
    
    # Input
    image_size: int = 224                # Kích thước ảnh đầu vào (224x224)
    num_frames: int = 16                 # Số frame lấy mẫu từ mỗi clip
    frame_sampling: str = "uniform"      # Cách lấy mẫu: "uniform" hoặc "random"
    
    # Face detection
    face_detector: str = "mtcnn"         # "mtcnn" hoặc "mediapipe"
    face_confidence: float = 0.95        # Ngưỡng tin cậy phát hiện khuôn mặt
    face_min_size: int = 60              # Kích thước khuôn mặt tối thiểu (pixel)


@dataclass
class AudioModuleConfig:
    """Cấu hình cho module Audio (CNN 1D + BiLSTM)."""
    
    # Model
    feature_dim: int = 512               # Kích thước vector đặc trưng đầu ra
    n_features: int = 120                # 40 MFCC + 40 delta + 40 delta-delta
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    lstm_hidden: int = 256               # Hidden size của BiLSTM
    lstm_layers: int = 2                 # Số layer LSTM
    bidirectional: bool = True           # BiLSTM hay LSTM thường
    dropout: float = 0.3
    
    # Audio processing
    sample_rate: int = 16000             # Tần số mẫu (Hz)
    n_mfcc: int = 40                     # Số hệ số MFCC
    n_fft: int = 2048                    # Kích thước FFT window
    hop_length: int = 512                # Bước nhảy giữa các frame
    max_duration: float = 15.0           # Thời lượng tối đa (giây)
    max_time_steps: int = 300            # Số time steps tối đa sau MFCC


@dataclass
class TextModuleConfig:
    """Cấu hình cho module Text (PhoBERT + BiLSTM)."""
    
    # Model
    feature_dim: int = 512               # Kích thước vector đặc trưng đầu ra
    phobert_dim: int = 768               # PhoBERT output dimension (fixed)
    lstm_hidden: int = 256               # Hidden size của BiLSTM
    lstm_layers: int = 2                 # Số layer LSTM
    bidirectional: bool = True           # BiLSTM
    dropout: float = 0.3
    freeze_phobert: bool = True          # GĐ1: freeze toàn bộ PhoBERT
    unfreeze_last_n: int = 0             # GĐ2: unfreeze N layers cuối
    
    # Text processing
    max_seq_length: int = 128            # Độ dài câu tối đa (số token)
    phobert_model: str = "vinai/phobert-base-v2"
    
    # Whisper STT
    whisper_model: str = "medium"        # "tiny", "base", "small", "medium", "large"
    whisper_language: str = "vi"         # Ngôn ngữ: tiếng Việt


@dataclass
class FusionConfig:
    """Cấu hình cho Fusion Hub."""
    
    # Fusion method
    method: str = "concat"               # "concat" (GĐ1) hoặc "cross_attention" (GĐ2)
    input_dim: int = 512                 # Kích thước feature từ mỗi module
    output_dim: int = 512                # Kích thước output
    num_modalities: int = 3              # Số phương thức (video, audio, text)
    
    # Cross-Attention (Giai đoạn 2)
    num_heads: int = 8                   # Số attention heads
    num_layers: int = 2                  # Số TransformerEncoder layers
    feedforward_dim: int = 2048          # FFN dimension trong Transformer
    attention_dropout: float = 0.1


@dataclass 
class TrainingConfig:
    """Cấu hình cho quá trình training."""
    
    # Hyperparameters cơ bản
    batch_size: int = 32
    learning_rate: float = 1e-4          # Tốc độ học
    weight_decay: float = 1e-5           # Regularization
    num_epochs: int = 50                 # Số epoch tối đa
    
    # Optimizer
    optimizer: str = "adamw"             # "adam", "adamw", "sgd"
    scheduler: str = "cosine"            # "cosine", "step", "plateau"
    warmup_steps: int = 500              # Số bước warmup
    
    # Early stopping
    patience: int = 10                   # Dừng sau N epoch không cải thiện
    min_delta: float = 0.001             # Mức cải thiện tối thiểu
    
    # Mixed precision (tận dụng A100)
    use_amp: bool = True                 # Automatic Mixed Precision
    
    # Gradient
    max_grad_norm: float = 1.0           # Gradient clipping
    accumulation_steps: int = 1          # Gradient accumulation
    
    # Reproducibility
    seed: int = 42
    
    # Logging
    log_interval: int = 10               # Log mỗi N step
    eval_interval: int = 1               # Evaluate mỗi N epoch


@dataclass
class EmotionLabels:
    """Nhãn cảm xúc."""
    
    # 7 cảm xúc cơ bản (Ekman + Neutral)
    labels: List[str] = field(default_factory=lambda: [
        "happy",        # Vui
        "sad",          # Buồn
        "angry",        # Giận
        "fear",         # Sợ
        "surprise",     # Ngạc nhiên
        "disgust",      # Ghê tởm
        "neutral",      # Trung tính
    ])
    
    # Tên tiếng Việt tương ứng
    labels_vi: List[str] = field(default_factory=lambda: [
        "Vui vẻ",
        "Buồn bã", 
        "Tức giận",
        "Sợ hãi",
        "Ngạc nhiên",
        "Ghê tởm",
        "Trung tính",
    ])
    
    @property
    def num_classes(self) -> int:
        return len(self.labels)
    
    def label_to_index(self, label: str) -> int:
        return self.labels.index(label.lower())
    
    def index_to_label(self, index: int, vietnamese: bool = False) -> str:
        if vietnamese:
            return self.labels_vi[index]
        return self.labels[index]


@dataclass
class ProjectConfig:
    """Cấu hình tổng hợp cho toàn bộ dự án."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    video: VideoModuleConfig = field(default_factory=VideoModuleConfig)
    audio: AudioModuleConfig = field(default_factory=AudioModuleConfig)
    text: TextModuleConfig = field(default_factory=TextModuleConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    emotions: EmotionLabels = field(default_factory=EmotionLabels)
    
    def setup(self):
        """Khởi tạo dự án: tạo thư mục, kiểm tra GPU, v.v."""
        self.paths.create_all()
        print(f"✅ Đã tạo cấu trúc thư mục tại: {PROJECT_ROOT}")
        print(f"📊 Số lớp cảm xúc: {self.emotions.num_classes}")
        print(f"🎯 Phương thức fusion: {self.fusion.method}")
        
        # Kiểm tra GPU
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("⚠️  Không tìm thấy GPU. Sẽ dùng CPU (chậm hơn nhiều).")


# ============================================
# Instance mặc định để import nhanh
# ============================================
config = ProjectConfig()


if __name__ == "__main__":
    # Test config
    config.setup()
    print(f"\n📁 Project root: {PROJECT_ROOT}")
    print(f"📁 Data root: {config.paths.data_root}")
    print(f"🎬 Video backbone: {config.video.backbone}")
    print(f"🔊 Audio MFCC: {config.audio.n_mfcc} coefficients")
    print(f"📝 Text max length: {config.text.max_seq_length}")
    print(f"🧠 Training epochs: {config.training.num_epochs}")
