"""Multimodal emotion analysis for Vietnamese video clips.

The service intentionally treats text as Vietnamese-first and visual/audio as
secondary evidence. Final labels are conservative: uncertain or conflicting
clips go to human review instead of being over-confidently auto-approved.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]

VI_LEXICON = {
    "happy": ["vui", "hạnh phúc", "cười", "thích", "yêu", "tuyệt", "may quá", "mừng", "sướng"],
    "sad": ["buồn", "khóc", "đau lòng", "cô đơn", "mất", "nhớ", "tủi", "thất vọng"],
    "angry": ["giận", "tức", "bực", "đồ khốn", "im đi", "câm", "ghét", "điên", "không tha"],
    "fear": ["sợ", "lo", "hoảng", "cứu", "nguy hiểm", "chạy đi", "đừng", "hãi"],
    "surprise": ["sao", "gì cơ", "thật á", "không thể", "bất ngờ", "trời ơi", "ủa"],
    "disgust": ["ghê", "kinh", "tởm", "bẩn", "khinh", "đáng ghét"],
}


class EmotionAnalyzer:
    def analyze_clip(self, face_images: Iterable[str] | None = None, transcript: str = "", audio_path: str | None = None) -> Dict[str, Any]:
        per_model: dict[str, dict[str, float]] = {}
        diagnostics: dict[str, Any] = {}

        visual = self._visual_emotion(face_images or [])
        if visual:
            per_model["visual_deepface"] = visual

        text = self._vietnamese_text_emotion(transcript or "")
        if text:
            per_model["text_vi_lexicon"] = text

        audio = self._audio_emotion(audio_path)
        if audio:
            per_model["audio_wav2vec"] = audio

        weights = {
            "visual_deepface": 0.45,
            "text_vi_lexicon": 0.35,
            "audio_wav2vec": 0.20,
        }
        combined = self._weighted_combine(per_model, weights)
        predicted = max(combined, key=combined.get) if combined else "unknown"
        confidence = float(combined.get(predicted, 0.0)) if combined else 0.0
        winners = {name: max(scores, key=scores.get) for name, scores in per_model.items() if scores}
        agreement_count = sum(1 for value in winners.values() if value == predicted)
        agreement = f"{agreement_count}/{len(winners)}" if winners else "0/0"
        diagnostics["model_winners"] = winners
        diagnostics["text_length"] = len((transcript or "").split())

        return {
            "predicted_emotion": predicted,
            "confidence": confidence,
            "agreement": agreement,
            "has_incongruity": len(set(winners.values())) > 1 if winners else False,
            "all_scores": combined or {emotion: 0.0 for emotion in EMOTIONS},
            "per_model_scores": per_model,
            "diagnostics": diagnostics,
        }

    def _visual_emotion(self, face_images: Iterable[str]) -> dict[str, float]:
        paths = [p for p in face_images if p and Path(p).exists()]
        if not paths:
            return {}
        aggregate = defaultdict(float)
        used = 0
        try:
            from backend.ai_models.model_manager import model_manager
            DeepFace = model_manager.load_model("deepface")
            for path in paths[:5]:
                result = DeepFace.analyze(path, actions=["emotion"], enforce_detection=False, silent=True)
                item = result[0] if isinstance(result, list) else result
                raw = item.get("emotion", {}) or {}
                normalized = self._normalize({self._map_emotion(k): float(v) for k, v in raw.items()})
                for emotion, score in normalized.items():
                    aggregate[emotion] += score
                used += 1
        except Exception as exc:
            print(f"⚠️ [EmotionAnalyzer] DeepFace visual emotion failed: {exc}")
            return {}
        if not used:
            return {}
        return self._normalize({emotion: aggregate[emotion] / used for emotion in EMOTIONS})

    def _vietnamese_text_emotion(self, transcript: str) -> dict[str, float]:
        text = transcript.lower().strip()
        if not text:
            return {}
        scores = {emotion: 0.0 for emotion in EMOTIONS}
        for emotion, phrases in VI_LEXICON.items():
            for phrase in phrases:
                if phrase in text:
                    scores[emotion] += 1.0 + min(1.0, len(phrase.split()) * 0.15)
        if sum(scores.values()) == 0:
            # Không ép neutral khi lời thoại không có tín hiệu cảm xúc rõ.
            return {}
        scores["neutral"] += 0.25
        return self._normalize(scores)

    def _audio_emotion(self, audio_path: str | None) -> dict[str, float]:
        if not audio_path or not Path(audio_path).exists():
            return {}
        try:
            from backend.ai_models.model_manager import model_manager
            classifier = model_manager.get_model("audio_emotion")
            if classifier is None:
                return {}
            result = classifier(audio_path)
            rows = result[0] if result and isinstance(result[0], list) else result
            return self._normalize({self._map_emotion(row.get("label", "")): float(row.get("score", 0.0)) for row in rows})
        except Exception:
            return {}

    def _weighted_combine(self, per_model: dict[str, dict[str, float]], weights: dict[str, float]) -> dict[str, float]:
        if not per_model:
            return {}
        acc = defaultdict(float)
        total_weight = 0.0
        for name, scores in per_model.items():
            weight = weights.get(name, 0.1)
            normalized = self._normalize(scores)
            for emotion, score in normalized.items():
                acc[emotion] += score * weight
            total_weight += weight
        if total_weight <= 0:
            return {"neutral": 1.0}
        return self._normalize({emotion: acc[emotion] / total_weight for emotion in EMOTIONS})

    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        mapped = {emotion: max(0.0, float(scores.get(emotion, 0.0))) for emotion in EMOTIONS}
        total = sum(mapped.values())
        if total <= 0:
            return {"neutral": 1.0}
        return {emotion: value / total for emotion, value in mapped.items()}

    @staticmethod
    def _map_emotion(label: str) -> str:
        label = (label or "").lower()
        if "happy" in label or "joy" in label or "happiness" in label:
            return "happy"
        if "sad" in label or "sadness" in label:
            return "sad"
        if "ang" in label:
            return "angry"
        if "fear" in label:
            return "fear"
        if "sur" in label:
            return "surprise"
        if "dis" in label:
            return "disgust"
        return "neutral"
