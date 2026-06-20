"""
MultimodalEngine — orchestrates all modality encoders and fusion.

Provides:
  process(text, image_path, audio_path) → MultimodalPayload
    .fused_embeds: torch.Tensor (B=1, seq_len, model_dim)  — for LLM.generate_from_embeds()
    .text_only_prompt: str  — fallback if fused inference unavailable
    .modalities_used: list[str]

Video handling:
  extract_frames(video_path, max_frames) → list[PIL.Image]
  Each frame goes through VisionEncoder; frames are averaged or concatenated.
"""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class ImageAnalyzer(Protocol):
    def analyze(self, image_path: str) -> dict[str, Any]:
        ...


class SpeechToTextEngine(Protocol):
    def transcribe(self, audio_path: str) -> str:
        ...


class TextToSpeechEngine(Protocol):
    def synthesize(self, text: str, output_path: str) -> str:
        ...


class TranscriptExtractor(Protocol):
    def fetch(self, youtube_url: str) -> str:
        ...


@dataclass(slots=True)
class MultimodalResult:
    source_type: str
    raw_text: str
    summary: str
    insights: list[str]
    metadata: dict[str, Any]


@dataclass(slots=True)
class MultimodalPayload:
    fused_embeds: Any
    text_only_prompt: str
    modalities_used: list[str] = field(default_factory=list)
    attention_mask: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ClipVitImageAnalyzer:
    def analyze(self, image_path: str) -> dict[str, Any]:
        tags = ["scene", "object", "visual-concept"]
        return {"image_path": image_path, "tags": tags, "model": "clip-vit-placeholder"}


class WhisperSTT:
    def transcribe(self, audio_path: str) -> str:
        return f"Transcribed speech from {audio_path}."


class SimpleTTS:
    def synthesize(self, text: str, output_path: str) -> str:
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(f"SYNTH_AUDIO::{text}")
        return output_path


class YouTubeTranscriptService:
    def fetch(self, youtube_url: str) -> str:
        return f"Transcript extracted from {youtube_url}."


class MultimodalEngine:
    def __init__(
        self,
        config: Any | None = None,
        llm_engine: Any | None = None,
        tokenizer: Any | None = None,
        image_analyzer: ImageAnalyzer | None = None,
        stt_engine: SpeechToTextEngine | None = None,
        tts_engine: TextToSpeechEngine | None = None,
        transcript_extractor: TranscriptExtractor | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or {}
        self.llm_engine = llm_engine
        self.tokenizer = tokenizer
        self.device = device

        # Backward-compatible lightweight utilities used by older routes/tests.
        self.image_analyzer = image_analyzer or ClipVitImageAnalyzer()
        self.stt_engine = stt_engine or WhisperSTT()
        self.tts_engine = tts_engine or SimpleTTS()
        self.transcript_extractor = transcript_extractor or YouTubeTranscriptService()

        self.vision_encoder = None
        self.audio_encoder = None
        self.fusion_layer = None
        if config is not None and llm_engine is not None and tokenizer is not None:
            self._build_native_stack()

    def process(
        self,
        text: str,
        image: Any | None = None,
        audio_path: str | None = None,
        video_path: str | None = None,
    ) -> MultimodalPayload:
        """Run the native text/image/audio/video fusion pipeline."""
        if self.fusion_layer is None:
            self._build_native_stack()

        torch = importlib.import_module("torch")
        modalities_used = ["text"]
        text_embeds = self._text_to_embeds(text)
        image_embeds = None
        audio_embeds = None
        metadata: dict[str, Any] = {}

        if image is not None:
            image_embeds = self.vision_encoder.encode(image).unsqueeze(0)
            modalities_used.append("image")

        if video_path is not None:
            frames = self.extract_video_frames(video_path)
            if frames:
                frame_embeds = self.vision_encoder.encode_batch(frames)
                image_embeds = frame_embeds.mean(dim=0, keepdim=True)
                modalities_used.append("video")
                metadata["video_frames"] = len(frames)

        if audio_path is not None:
            audio_embeds = self.audio_encoder.encode(audio_path).unsqueeze(0)
            modalities_used.append("audio")

        fused = self.fusion_layer(text_embeds, image_embeds, audio_embeds)
        attention_mask = torch.ones(
            (fused.shape[0], fused.shape[1]),
            dtype=torch.long,
            device=fused.device,
        )
        return MultimodalPayload(
            fused_embeds=fused,
            text_only_prompt=self._fallback_prompt(text, modalities_used),
            modalities_used=modalities_used,
            attention_mask=attention_mask,
            metadata=metadata,
        )

    def extract_video_frames(self, video_path: str, max_frames: int | None = None) -> list[Any]:
        """Sample frames from a video using cv2.VideoCapture."""
        cv2 = importlib.import_module("cv2")
        image_module = importlib.import_module("PIL.Image")
        video_cfg = self._cfg("multimodal.video", {})
        frame_sample_rate = int(self._cfg("multimodal.video.frame_sample_rate", 1))
        max_frames = max_frames or int(video_cfg.get("max_frames", 8))

        capture = cv2.VideoCapture(video_path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 1
        stride = max(1, int(fps * frame_sample_rate))
        frames: list[Any] = []
        frame_index = 0
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(image_module.fromarray(rgb_frame))
            frame_index += 1
        capture.release()
        return frames

    def _text_to_embeds(self, text: str) -> Any:
        """Tokenize text and map token ids through the LLM input embedding layer."""
        torch = importlib.import_module("torch")
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        model = self._llm_model()
        embeddings = model.get_input_embeddings()
        with torch.no_grad():
            return embeddings(input_ids.to(embeddings.weight.device))

    def fine_tune_projections(self, dataset_path: str, epochs: int) -> None:
        """Train only projection and fusion parameters from JSONL/.json/.pt examples.

        Expected examples can contain text plus optional image/audio/video paths. If a
        `target_embeds_path` or `target_embeds` tensor is provided, mean-squared error is
        optimized between fused embeddings and the target. Base encoders and LLM
        backbone remain frozen.
        """
        if self.fusion_layer is None:
            self._build_native_stack()

        torch = importlib.import_module("torch")
        trainable_modules = [self.fusion_layer]
        if self.vision_encoder is not None:
            trainable_modules.extend([self.vision_encoder.norm, self.vision_encoder.projection])
        if self.audio_encoder is not None:
            trainable_modules.extend([self.audio_encoder.norm, self.audio_encoder.projection])

        params = [param for module in trainable_modules for param in module.parameters()]
        optimizer = torch.optim.AdamW(params, lr=float(self._cfg("training.learning_rate", 0.0002)))
        examples = self._load_finetune_examples(dataset_path)
        for _ in range(epochs):
            for example in examples:
                target = self._load_target_embeds(example)
                if target is None:
                    continue
                payload = self.process(
                    example.get("text", ""),
                    image=example.get("image") or example.get("image_path"),
                    audio_path=example.get("audio_path"),
                    video_path=example.get("video_path"),
                )
                loss = torch.nn.functional.mse_loss(payload.fused_embeds, target.to(payload.fused_embeds.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def summarize(self, text: str, max_sentences: int = 2) -> str:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return ""
        return " ".join(sentences[:max_sentences])

    def extract_insights(self, text: str) -> list[str]:
        insights: list[str] = []
        lower = text.lower()
        if any(k in lower for k in ["increase", "increased", "growth", "improve"]):
            insights.append("Positive trend detected")
        if any(k in lower for k in ["risk", "issue", "error", "decline"]):
            insights.append("Potential risk detected")
        numbers = re.findall(r"\d+(?:\.\d+)?%?", text)
        if numbers:
            insights.append(f"Key metrics referenced: {', '.join(numbers[:3])}")
        if not insights:
            insights.append("General informational content")
        return insights

    def process_image(self, image_path: str) -> MultimodalResult:
        analysis = self.image_analyzer.analyze(image_path)
        text = f"Image tags: {', '.join(analysis.get('tags', []))}."
        return MultimodalResult(
            source_type="image",
            raw_text=text,
            summary=self.summarize(text),
            insights=self.extract_insights(text),
            metadata=analysis,
        )

    def process_video(self, video_path: str) -> MultimodalResult:
        transcript = self.stt_engine.transcribe(video_path)
        return MultimodalResult(
            source_type="video",
            raw_text=transcript,
            summary=self.summarize(transcript),
            insights=self.extract_insights(transcript),
            metadata={"video_path": video_path, "pipeline": "video->audio->whisper"},
        )

    def process_speech(self, audio_path: str) -> MultimodalResult:
        transcript = self.stt_engine.transcribe(audio_path)
        return MultimodalResult(
            source_type="speech",
            raw_text=transcript,
            summary=self.summarize(transcript),
            insights=self.extract_insights(transcript),
            metadata={"audio_path": audio_path, "model": "whisper-placeholder"},
        )

    def text_to_speech(self, text: str, output_path: str) -> str:
        return self.tts_engine.synthesize(text, output_path)

    def process_youtube(self, youtube_url: str) -> MultimodalResult:
        transcript = self.transcript_extractor.fetch(youtube_url)
        return MultimodalResult(
            source_type="youtube",
            raw_text=transcript,
            summary=self.summarize(transcript),
            insights=self.extract_insights(transcript),
            metadata={"youtube_url": youtube_url},
        )

    def _build_native_stack(self) -> None:
        VisionEncoder = importlib.import_module("app.multimodal.vision_encoder").VisionEncoder
        AudioEncoder = importlib.import_module("app.multimodal.audio_encoder").AudioEncoder
        MultimodalFusionLayer = importlib.import_module(
            "app.multimodal.fusion_layer"
        ).MultimodalFusionLayer

        model_dim = int(self._cfg("model.architecture.dim", 3072))
        image_cfg = self._cfg("multimodal.image", {})
        audio_cfg = self._cfg("multimodal.audio", {})
        fusion_cfg = self._cfg("multimodal.fusion", {})

        self.vision_encoder = VisionEncoder(
            clip_model_id=image_cfg.get("encoder", "openai/clip-vit-large-patch14"),
            encoder_dim=int(image_cfg.get("encoder_dim", 1024)),
            projection_dim=int(image_cfg.get("projection_dim", model_dim)),
            device=self.device,
        )
        self.audio_encoder = AudioEncoder(
            whisper_size=audio_cfg.get("whisper_size", "base"),
            encoder_dim=int(audio_cfg.get("encoder_dim", 512)),
            projection_dim=int(audio_cfg.get("projection_dim", model_dim)),
            device=self.device,
        )
        self.fusion_layer = MultimodalFusionLayer(
            model_dim=model_dim,
            fusion_mode=self._cfg("multimodal.fusion_mode", "cross_attention"),
            n_heads=int(fusion_cfg.get("n_heads", 12)),
            n_layers=int(fusion_cfg.get("n_cross_attention_layers", 2)),
        )

        vision_checkpoint = image_cfg.get("projection_checkpoint")
        if vision_checkpoint and Path(vision_checkpoint).exists():
            self.vision_encoder.load_projection(vision_checkpoint)
        audio_checkpoint = audio_cfg.get("projection_checkpoint")
        if audio_checkpoint and Path(audio_checkpoint).exists():
            self.audio_encoder.load_projection(audio_checkpoint)
        fusion_checkpoint = fusion_cfg.get("checkpoint")
        if fusion_checkpoint and Path(fusion_checkpoint).exists():
            self.fusion_layer.load(fusion_checkpoint)

    def _cfg(self, path: str, default: Any = None) -> Any:
        current = self.config
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part, default)
            else:
                current = getattr(current, part, default)
            if current is default:
                return default
        return current

    def _llm_model(self) -> Any:
        for attr in ("model", "_hf_model", "_model"):
            model = getattr(self.llm_engine, attr, None)
            if model is not None:
                return model
        if hasattr(self.llm_engine, "_ensure_transformers_loaded"):
            self.llm_engine._ensure_transformers_loaded()
            model = getattr(self.llm_engine, "_hf_model", None)
            if model is not None:
                return model
        raise RuntimeError("LLM model with get_input_embeddings() is not available")

    def _fallback_prompt(self, text: str, modalities_used: list[str]) -> str:
        extra = [modality for modality in modalities_used if modality != "text"]
        if not extra:
            return text
        return f"{text}\n\nAttached modalities: {', '.join(extra)}."

    def _load_finetune_examples(self, dataset_path: str) -> list[dict[str, Any]]:
        path = Path(dataset_path)
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else data.get("examples", [])
        if path.suffix in {".pt", ".pth"}:
            torch = importlib.import_module("torch")
            data = torch.load(path, map_location="cpu")
            return data if isinstance(data, list) else data.get("examples", [])
        raise ValueError(f"Unsupported dataset format: {dataset_path}")

    def _load_target_embeds(self, example: dict[str, Any]) -> Any | None:
        torch = importlib.import_module("torch")
        if "target_embeds" in example:
            return torch.tensor(example["target_embeds"])
        if "target_embeds_path" in example:
            return torch.load(example["target_embeds_path"], map_location="cpu")
        return None
