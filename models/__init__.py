"""
Models package - Các module mô hình cho hệ thống phân tích cảm xúc đa phương thức.

Cách dùng:
    from models import VideoModule, AudioModule, TextModule, MultimodalClassifier
"""

from models.video_module import VideoModule
from models.audio_module import AudioModule
from models.text_module import TextModule
from models.fusion import FusionHub, ConcatFusion, CrossAttentionFusion
from models.classifier import MultimodalClassifier

__all__ = [
    "VideoModule",
    "AudioModule", 
    "TextModule",
    "FusionHub",
    "ConcatFusion",
    "CrossAttentionFusion",
    "MultimodalClassifier",
]
