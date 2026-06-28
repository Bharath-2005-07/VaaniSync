"""
Unit tests for the Video Localizer pipeline — video_localizer/agent.py
All heavy deps (ffmpeg, faster-whisper, MeloTTS, pydub, Ollama) are mocked.
No real files or network calls are made.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("PATH", "")

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from video_localizer.agent import (  # noqa: E402
    ExtractionResult,
    MuxResult,
    PipelineInput,
    SynthesisResult,
    TranscriptionResult,
    TranslationResult,
    _detect_gender_heuristics,
    _pad_or_trim,
    _translate_batch,
    extract_audio,
    mux_video,
    parse_input,
    synthesise_segments,
    transcribe_audio,
    translate_segments,
)


class MockContext:
    def __init__(self, state=None):
        self.state = state if state is not None else {}


def _collect_events(generator):
    return list(generator)


# ---------------------------------------------------------------------------
# Stage 0: Parse Input
# ---------------------------------------------------------------------------


def test_parse_input_string():
    ctx = MockContext()
    events = _collect_events(parse_input(ctx, "Dub video/video1.mp4 into Kannada female"))

    assert len(events) == 2
    assert ctx.state["video_path"] == "video/video1.mp4"
    assert ctx.state["target_language"] == "Kannada"
    assert ctx.state["speaker_gender"] == "female"

    output = events[-1].output
    assert isinstance(output, PipelineInput)
    assert output.speaker_gender == "female"


def test_parse_input_object():
    ctx = MockContext()
    inp = PipelineInput(video_path="video/custom.mp4", target_language="Kannada", speaker_gender="male")
    events = _collect_events(parse_input(ctx, inp))

    assert len(events) == 2
    assert ctx.state["video_path"] == "video/custom.mp4"
    assert ctx.state["speaker_gender"] == "male"


def test_parse_input_dict():
    ctx = MockContext()
    inp = {"video_path": "video/dict.mp4", "speaker_gender": "male"}
    events = _collect_events(parse_input(ctx, inp))

    assert len(events) == 2
    assert ctx.state["video_path"] == "video/dict.mp4"
    assert ctx.state["speaker_gender"] == "male"


# ---------------------------------------------------------------------------
# Stage 1: Audio Extraction
# ---------------------------------------------------------------------------


@patch("subprocess.run")
def test_extract_audio(mock_run):
    ctx = MockContext()
    inp = PipelineInput(video_path="video/video1.mp4", speaker_gender="auto")

    events = _collect_events(extract_audio(ctx, inp))

    assert len(events) == 2
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert "ffmpeg" in cmd
    assert "-af" in cmd
    assert "afftdn" in cmd

    output = events[-1].output
    assert isinstance(output, ExtractionResult)
    assert output.audio_path == "audio/original_audio.wav"


# ---------------------------------------------------------------------------
# Stage 2: Transcription
# ---------------------------------------------------------------------------


class MockWhisperSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


@patch("faster_whisper.WhisperModel")
@patch("video_localizer.agent.Path.write_text")
def test_transcribe_audio(mock_write, mock_whisper_cls):
    mock_model = MagicMock()
    mock_whisper_cls.return_value = mock_model

    # Mock return values for transcribe()
    mock_segments = [
        MockWhisperSegment(0.0, 1.5, "Hello world"),
        MockWhisperSegment(1.5, 3.0, "This is a test"),
    ]
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_model.transcribe.return_value = (mock_segments, mock_info)

    ctx = MockContext()
    inp = ExtractionResult(video_path="video/video1.mp4", audio_path="audio/original_audio.wav")

    events = _collect_events(transcribe_audio(ctx, inp))

    assert len(events) == 2
    mock_model.transcribe.assert_called_once_with(
        "audio/original_audio.wav",
        beam_size=5,
        word_timestamps=False,
        condition_on_previous_text=False,
    )

    output = events[-1].output
    assert isinstance(output, TranscriptionResult)
    assert output.segment_count == 2
    assert output.detected_language == "en"
    mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# Stage 3: Translation
# ---------------------------------------------------------------------------


def test_detect_gender_heuristics():
    segs = [
        {"text": "He is a good man"},
        {"text": "She went to school"},
        {"text": "Hello world"},
    ]
    genders = _detect_gender_heuristics(segs)
    assert genders == ["male", "female", "female"]


@patch("deep_translator.GoogleTranslator.translate")
def test_translate_batch_google(mock_translate):
    mock_translate.return_value = "ಕನ್ನಡ 1 ||| ಕನ್ನಡ 2"

    segs = [{"text": "Hello 1"}, {"text": "Hello 2"}]
    res = _translate_batch(segs)

    assert len(res) == 2
    assert res[0]["text"] == "ಕನ್ನಡ 1"
    assert res[1]["text"] == "ಕನ್ನಡ 2"


@patch("deep_translator.GoogleTranslator.translate")
def test_translate_batch_fallback(mock_translate):
    mock_translate.side_effect = Exception("Service unavailable")

    segs = [{"text": "Hello 1"}, {"text": "Hello 2"}]
    res = _translate_batch(segs)

    # Fallback should return original text
    assert len(res) == 2
    assert res[0]["text"] == "Hello 1"
    assert res[1]["text"] == "Hello 2"


@patch("video_localizer.agent.Path.read_text")
@patch("video_localizer.agent.Path.write_text")
@patch("video_localizer.agent._translate_batch")
def test_translate_segments(mock_translate_batch, mock_write, mock_read):
    mock_read.return_value = json.dumps([
        {"start": 0.0, "end": 2.0, "text": "Hello"},
        {"start": 2.0, "end": 4.0, "text": "World"}
    ])

    mock_translate_batch.return_value = [
        {"id": 1, "text": "ಕನ್ನಡ 1"},
        {"id": 2, "text": "ಕನ್ನಡ 2"}
    ]

    ctx = MockContext(state={"speaker_gender": "auto"})
    inp = TranscriptionResult(
        audio_path="audio/original_audio.wav",
        segments_path="transcripts/segments.json",
        segment_count=2,
        detected_language="en",
    )

    old_key = os.environ.get("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = "fake-api-key"
    try:
        events = _collect_events(translate_segments(ctx, inp))
    finally:
        if old_key is None:
            del os.environ["GOOGLE_API_KEY"]
        else:
            os.environ["GOOGLE_API_KEY"] = old_key

    assert len(events) == 2
    output = events[-1].output
    assert isinstance(output, TranslationResult)
    assert output.segment_count == 2
    mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# Stage 4: Synthesis
# ---------------------------------------------------------------------------


@patch("video_localizer.agent._change_speed_ffmpeg")
@patch("pydub.AudioSegment.from_wav")
def test_pad_or_trim_speedup(mock_from_wav, mock_change_speed):
    mock_audio = MagicMock()
    mock_audio.set_frame_rate.return_value = mock_audio
    mock_audio.set_channels.return_value = mock_audio
    mock_audio.__len__.return_value = 2000
    mock_audio.frame_rate = 16000
    mock_from_wav.return_value = mock_audio

    # Target duration is 1000ms (so ratio = 2000 / 1000 = 2.0 -> speedup)
    _pad_or_trim("audio/dummy.wav", 1000)
    mock_change_speed.assert_called_once_with("audio/dummy.wav", 2.0)


@patch("video_localizer.agent.Path.read_text")
@patch("video_localizer.agent._pad_or_trim")
@patch("os.path.exists")
@patch("pydub.AudioSegment.from_mp3")
@patch("pydub.AudioSegment.silent")
def test_synthesise_segments_edge_tts(mock_silent, mock_from_mp3, mock_exists, mock_pad_or_trim, mock_read):
    mock_read.return_value = json.dumps([
        {"start": 0.0, "end": 2.0, "text": "ಕನ್ನಡ 1", "gender": "female"},
        {"start": 2.0, "end": 4.0, "text": "ಕನ್ನಡ 2", "gender": "male"}
    ])
    mock_exists.return_value = True

    ctx = MockContext()
    inp = TranslationResult(segments_path="transcripts/translated_segments.json", segment_count=2)

    # We mock asyncio.run / edge_tts to simulate success
    with patch("asyncio.run") as mock_async_run:
        events = _collect_events(synthesise_segments(ctx, inp))

        assert len(events) == 2
        output = events[-1].output
        assert isinstance(output, SynthesisResult)
        assert output.segment_count == 2
        # asyncio.run is called within ThreadPoolExecutor workers
        assert mock_async_run.call_count == 2


# ---------------------------------------------------------------------------
# Stage 5: Muxing
# ---------------------------------------------------------------------------


@patch("subprocess.run")
@patch("video_localizer.agent.Path.read_text")
@patch("pydub.AudioSegment.silent")
@patch("pydub.AudioSegment.from_wav")
def test_mux_video(mock_from_wav, mock_silent, mock_read, mock_run):
    mock_read.return_value = json.dumps([
        {"start": 0.0, "end": 2.0, "text": "ಕನ್ನಡ 1"},
        {"start": 2.0, "end": 4.0, "text": "ಕನ್ನಡ 2"}
    ])

    mock_audio = MagicMock()
    mock_silent.return_value = mock_audio
    mock_from_wav.return_value = mock_audio

    ctx = MockContext(state={"video_path": "video/video1.mp4"})
    inp = SynthesisResult(segments_dir="audio/dubbed_segments", segment_count=2)

    events = _collect_events(mux_video(ctx, inp))

    assert len(events) == 2
    output = events[-1].output
    assert isinstance(output, MuxResult)
    assert Path(output.output_path).as_posix() == "output/video1.mp4"
    mock_run.assert_called_once()
