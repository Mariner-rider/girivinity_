from pathlib import Path

from multimodal_engine import MultimodalEngine


def test_image_video_speech_and_youtube_processing():
    engine = MultimodalEngine()

    image_result = engine.process_image("demo.png")
    assert image_result.source_type == "image"
    assert "Image tags" in image_result.raw_text

    video_result = engine.process_video("demo.mp4")
    assert video_result.source_type == "video"
    assert "Transcribed speech" in video_result.raw_text

    speech_result = engine.process_speech("voice.wav")
    assert speech_result.source_type == "speech"

    yt_result = engine.process_youtube("https://youtube.com/watch?v=abc")
    assert yt_result.source_type == "youtube"
    assert "Transcript extracted" in yt_result.raw_text


def test_tts_summarization_and_insight_extraction(tmp_path: Path):
    engine = MultimodalEngine()
    out = tmp_path / "speech.txt"
    output_path = engine.text_to_speech("Revenue increased 20% but there is risk.", str(out))

    assert Path(output_path).exists()

    summary = engine.summarize("First sentence. Second sentence. Third sentence.", max_sentences=2)
    assert summary == "First sentence. Second sentence."

    insights = engine.extract_insights("Revenue increased 20% but there is risk.")
    assert "Positive trend detected" in insights
    assert "Potential risk detected" in insights
    assert any("20%" in insight for insight in insights)
