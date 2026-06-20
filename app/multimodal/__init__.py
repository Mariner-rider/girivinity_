from app.multimodal.processor import MultimodalPayload as ProcessorPayload
from app.multimodal.processor import MultimodalProcessor

__all__ = [
    "AudioEncoder",
    "MultimodalFusionLayer",
    "MultimodalProcessor",
    "ProcessorPayload",
    "VisionEncoder",
]


def __getattr__(name: str):
    if name == "VisionEncoder":
        from app.multimodal.vision_encoder import VisionEncoder

        return VisionEncoder
    if name == "AudioEncoder":
        from app.multimodal.audio_encoder import AudioEncoder

        return AudioEncoder
    if name == "MultimodalFusionLayer":
        from app.multimodal.fusion_layer import MultimodalFusionLayer

        return MultimodalFusionLayer
    raise AttributeError(name)
