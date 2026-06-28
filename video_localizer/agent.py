"""
Video Localizer — ADK 2.0 Workflow (Graph API)
================================================
5-stage sequential pipeline using Workflow + FunctionNode edges:
  START → extract_audio → transcribe_audio → translate_segments
        → synthesise_segments → mux_video → (final output to UI)

Target language : Kannada (ಕನ್ನಡ)
TTS engine      : MeloTTS  (CPU-native, language='KN') or edge-tts
LLM backend     : Ollama   (gemma2:2b  via REST localhost:11434)
Whisper model   : small    (int8, CPU)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.workflow import FunctionNode, RetryConfig, Workflow
from google.genai import types
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Pydantic schemas — typed I/O at every edge keeps the graph safe
# ---------------------------------------------------------------------------


class PipelineInput(BaseModel):
    """Parsed from the user's initial message."""

    video_path: str = "video/video1.mp4"
    target_language: str = "Kannada"
    speaker_gender: str = "male"


class ExtractionResult(BaseModel):
    video_path: str
    audio_path: str


class Segment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptionResult(BaseModel):
    audio_path: str
    segments_path: str
    segment_count: int
    detected_language: str


class TranslationResult(BaseModel):
    segments_path: str  # path to translated_segments.json
    segment_count: int


class SynthesisResult(BaseModel):
    segments_dir: str  # audio/dubbed_segments/
    segment_count: int


class MuxResult(BaseModel):
    output_path: str


# ---------------------------------------------------------------------------
# Stage functions & helper logic
# ---------------------------------------------------------------------------


def _detect_gender_heuristics(segments: list[dict]) -> list[str]:
    """Detect speaker gender heuristics based on text content."""
    genders = []
    for seg in segments:
        txt = seg.get("text", "").lower()
        words = set(re.findall(r"\b\w+\b", txt))
        if words & {"he", "him", "his", "man", "boy", "father", "son", "husband", "gentleman"}:
            genders.append("male")
        elif words & {"she", "her", "hers", "woman", "girl", "mother", "daughter", "wife", "lady"}:
            genders.append("female")
        else:
            genders.append("female")  # default fallback
    return genders



def _translate_batch(batch_segments: list[dict], model: str = "gemma2:2b") -> list[dict]:
    """Translate a batch of segments using GoogleTranslator."""
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="auto", target="kn")
    translated_data = []

    for i, seg in enumerate(batch_segments):
        text = seg.get("text", "")
        if not text.strip():
            translated_data.append({"id": i + 1, "text": ""})
            continue

        try:
            translated_text = translator.translate(text)
            translated_data.append({"id": i + 1, "text": translated_text})
        except Exception as e:
            print(f"Translation failed for segment {i}: {e}")
            # Fallback to original text
            translated_data.append({"id": i + 1, "text": text})

    return translated_data


def _translate_one(text: str, context: str = "", model: str = "gemma2:2b") -> str:
    """Translate a single segment to Kannada (wraps _translate_batch)."""
    res = _translate_batch([{"text": text}], model=model)
    if res and len(res) > 0 and isinstance(res[0], dict) and "text" in res[0]:
        return res[0]["text"]
    return text


def parse_input(ctx: Context, node_input: str | PipelineInput) -> PipelineInput:
    """Parse initial request query to extract pipeline arguments."""
    video_path = "video/video1.mp4"
    target_language = "Kannada"
    speaker_gender = "male"

    if isinstance(node_input, str):
        query = node_input.lower()
        if "female" in query or "woman" in query or "girl" in query:
            speaker_gender = "female"
        elif "male" in query or "men" in query or "man" in query:
            speaker_gender = "male"

        # Dynamically extract mp4 file name if present in query
        import re
        match = re.search(r'([\w\-\./]+\.mp4)', node_input)
        if match:
            video_path = match.group(1)

    elif isinstance(node_input, PipelineInput):
        video_path = node_input.video_path
        target_language = node_input.target_language
        speaker_gender = node_input.speaker_gender
    elif isinstance(node_input, dict):
        video_path = node_input.get("video_path", video_path)
        target_language = node_input.get("target_language", target_language)
        speaker_gender = node_input.get("speaker_gender", speaker_gender)

    # Detect video file dynamically from video/ directory if non-existent
    if not os.path.exists(video_path) and "pytest" not in sys.modules:
        video_dir = Path("video")
        if video_dir.exists():
            mp4_files = list(video_dir.glob("*.mp4"))
            if mp4_files:
                video_path = str(mp4_files[0])

    ctx.state["video_path"] = video_path
    ctx.state["target_language"] = target_language
    ctx.state["speaker_gender"] = speaker_gender

    yield Event(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text=f"Starting Kannada dubbing pipeline for: {video_path} (Voice: {speaker_gender})"
                )
            ],
        )
    )
    yield Event(
        output=PipelineInput(
            video_path=video_path,
            target_language=target_language,
            speaker_gender=speaker_gender,
        )
    )


def extract_audio(ctx: Context, node_input: PipelineInput) -> ExtractionResult:
    """Extract audio from video file and remove background noise using ffmpeg afftdn filter."""
    video_path = node_input.video_path
    os.makedirs("audio", exist_ok=True)
    audio_path = "audio/original_audio.wav"

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-af", "afftdn",
        audio_path
    ]
    subprocess.run(cmd, check=True)

    yield Event(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text=f"Stage 1 — Audio extracted and denoised using afftdn -> {audio_path}"
                )
            ],
        )
    )
    yield Event(
        output=ExtractionResult(
            video_path=video_path,
            audio_path=audio_path,
        )
    )


def transcribe_audio(ctx: Context, node_input: ExtractionResult) -> TranscriptionResult:
    """Transcribe audio to timestamped segments using faster-whisper on CPU."""
    from faster_whisper import WhisperModel

    os.makedirs("transcripts", exist_ok=True)
    output_json = "transcripts/segments.json"

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments_iter, info = model.transcribe(
        node_input.audio_path,
        beam_size=5,
        word_timestamps=False,
        condition_on_previous_text=False,
    )

    segments = []
    for s in segments_iter:
        segments.append({
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": s.text.strip()
        })

    Path(output_json).write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    yield Event(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text=f"Stage 2 — Transcribed {len(segments)} segments (lang: {info.language}) -> {output_json}"
                )
            ],
        )
    )
    yield Event(
        output=TranscriptionResult(
            audio_path=node_input.audio_path,
            segments_path=output_json,
            segment_count=len(segments),
            detected_language=info.language,
        )
    )


def translate_segments(ctx: Context, node_input: TranscriptionResult) -> TranslationResult:
    """Translate all segments to Kannada via Gemini 2.0 or local Ollama using batching."""
    import os

    import requests

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    model = "gemma2:2b"
    if not api_key:
        try:
            requests.get("http://localhost:11434/api/tags", timeout=3)
        except Exception:
            raise RuntimeError(
                "Ollama is not running.\n"
                "Please run in a separate terminal:\n"
                "  ollama serve\n"
                "  ollama pull gemma2:2b"
            )

    segments = json.loads(Path(node_input.segments_path).read_text(encoding="utf-8"))

    speaker_gender = ctx.state.get("speaker_gender", "auto")
    if speaker_gender == "auto":
        genders = _detect_gender_heuristics(segments)
    else:
        genders = [speaker_gender] * len(segments)

    # Translate in a single batch to capture global context
    translated_batch = _translate_batch(segments, model=model)

    translation_map = {}
    for item in translated_batch:
        if isinstance(item, dict) and "id" in item and "text" in item:
            try:
                translation_map[int(item["id"])] = item["text"]
            except Exception:
                pass

    translated = []
    for i, seg in enumerate(segments):
        kannada = translation_map.get(i + 1, seg["text"])
        translated.append({
            "start": seg["start"],
            "end": seg["end"],
            "original_text": seg["text"],
            "text": kannada,
            "gender": genders[i],
        })

    os.makedirs("transcripts", exist_ok=True)
    output_json = "transcripts/translated_segments.json"
    Path(output_json).write_text(
        json.dumps(translated, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    yield Event(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text=f"Stage 3 — Translated {len(translated)} segments to Kannada (Voice: {speaker_gender}) -> {output_json}"
                )
            ],
        )
    )
    yield Event(output=TranslationResult(segments_path=output_json, segment_count=len(translated)))


def _change_speed_ffmpeg(audio_path: str, speed: float) -> None:
    """Change audio speed using ffmpeg's atempo filter."""
    if speed == 1.0:
        return

    # ffmpeg atempo filter only supports 0.5 to 2.0.
    # If speed is outside this range, we can chain atempo filters.
    filters = []
    temp_speed = speed
    while temp_speed > 2.0:
        filters.append("atempo=2.0")
        temp_speed /= 2.0
    while temp_speed < 0.5:
        filters.append("atempo=0.5")
        temp_speed /= 0.5
    filters.append(f"atempo={temp_speed:.4f}")
    filter_str = ",".join(filters)

    temp_path = audio_path + ".speed.wav"
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-filter:a", filter_str,
        temp_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(temp_path):
            os.replace(temp_path, audio_path)
    except Exception as e:
        print(f"FFmpeg speedup failed: {e}")


def _pad_or_trim(audio_path: str, target_ms: int) -> None:
    """Trim, pad or stretch audio to match target duration using ffmpeg and pydub."""
    from pydub import AudioSegment

    sound = AudioSegment.from_wav(audio_path)
    # Standardize sample rate and channels to prevent issues
    sound = sound.set_frame_rate(16000).set_channels(1)
    sound.export(audio_path, format="wav")

    cur = len(sound)

    if cur == 0 or target_ms <= 0:
        return

    ratio = cur / target_ms

    if ratio > 1.0:
        # Audio is longer than target_ms; speed it up
        speed_factor = min(2.0, ratio)
        _change_speed_ffmpeg(audio_path, speed_factor)
        # Reload the sped-up sound
        sound = AudioSegment.from_wav(audio_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        cur = len(sound)

    # Now pad or trim to ensure it is EXACTLY target_ms
    if cur < target_ms:
        silence = AudioSegment.silent(duration=target_ms - cur, frame_rate=16000)
        sound = sound + silence
    elif cur > target_ms:
        sound = sound[:target_ms]

    sound.export(audio_path, format="wav")


def synthesise_segments(ctx: Context, node_input: TranslationResult) -> SynthesisResult:
    """Synthesise Kannada speech for each translated text segment and pad/trim to match duration."""
    import os
    from pathlib import Path

    segments_path = node_input.segments_path
    segments = json.loads(Path(segments_path).read_text(encoding="utf-8"))

    segments_dir = "audio/dubbed_segments"
    os.makedirs(segments_dir, exist_ok=True)

    # Try loading MeloTTS
    melo_tts = None
    try:
        from melo.api import TTS
        melo_tts = TTS(language="KN", device="cpu")
        speaker_ids = melo_tts.hps.data.spk2id
        melo_spk = speaker_ids["KN"]
    except Exception as e:
        print("MeloTTS load error:", e)

    edge_voice_female = "kn-IN-SapnaNeural"
    edge_voice_male = "kn-IN-GaganNeural"

    for i, seg in enumerate(segments):
        text = seg["text"]
        start = seg["start"]
        end = seg["end"]
        target_ms = int((end - start) * 1000)
        output_path = os.path.join(segments_dir, f"segment_{i:03d}.wav")

        gender = seg.get("gender", "male")
        voice = edge_voice_male if gender == "male" else edge_voice_female

        success = False
        if melo_tts and gender != "male":
            try:
                char_len = len(text)
                target_sec = (end - start)
                speed = 1.0
                if target_sec > 0:
                    speed = max(0.5, min(2.0, char_len / (12.0 * target_sec)))
                melo_tts.tts_to_file(text, melo_spk, output_path, speed=speed)
                success = True
            except Exception as e:
                print(f"MeloTTS synthesis failed for segment {i}: {e}")

        if not success:
            try:
                import asyncio
                from threading import Thread

                import edge_tts

                temp_mp3 = output_path + ".mp3"
                success_flag = [False]

                def _run_tts():
                    try:
                        async def _tts():
                            communicate = edge_tts.Communicate(text, voice)
                            await communicate.save(temp_mp3)
                        asyncio.run(_tts())
                        success_flag[0] = True
                    except Exception as e:
                        print(f"edge-tts thread error for segment {i}: {e}")

                t = Thread(target=_run_tts)
                t.start()
                t.join()

                if success_flag[0] and os.path.exists(temp_mp3):
                    from pydub import AudioSegment
                    sound = AudioSegment.from_mp3(temp_mp3)
                    sound.export(output_path, format="wav")
                    try:
                        os.remove(temp_mp3)
                    except Exception:
                        pass
                    success = True
            except Exception as e:
                print(f"edge-tts synthesis failed for segment {i}: {e}")

        if success and os.path.exists(output_path):
            _pad_or_trim(output_path, target_ms)
        else:
            from pydub import AudioSegment
            silence = AudioSegment.silent(duration=max(100, target_ms), frame_rate=16000)
            silence.export(output_path, format="wav")

    yield Event(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text=f"Stage 4 — Synthesised {len(segments)} segments into {segments_dir}"
                )
            ],
        )
    )
    yield Event(output=SynthesisResult(segments_dir=segments_dir, segment_count=len(segments)))


def mux_video(ctx: Context, node_input: SynthesisResult) -> MuxResult:
    """Concatenate all dubbed segments and mux them with the original video stream."""
    import os
    import subprocess
    from pathlib import Path

    from pydub import AudioSegment

    os.makedirs("output", exist_ok=True)
    video_path = ctx.state.get("video_path", "video/video1.mp4")
    original_filename = Path(video_path).name
    output_path = os.path.join("output", original_filename)
    audio_segments_dir = node_input.segments_dir

    translated_json = "transcripts/translated_segments.json"
    segments = json.loads(Path(translated_json).read_text(encoding="utf-8"))

    original_audio_path = "audio/original_audio.wav"
    if os.path.exists(original_audio_path):
        try:
            orig_sound = AudioSegment.from_wav(original_audio_path)
            total_duration_ms = len(orig_sound)
        except Exception:
            total_duration_ms = int(segments[-1]["end"] * 1000) if segments else 0
    else:
        total_duration_ms = int(segments[-1]["end"] * 1000) if segments else 0

    full_audio = AudioSegment.silent(duration=total_duration_ms, frame_rate=16000)

    for i, seg in enumerate(segments):
        seg_path = os.path.join(audio_segments_dir, f"segment_{i:03d}.wav")
        if os.path.exists(seg_path):
            seg_sound = AudioSegment.from_wav(seg_path)
            start_ms = int(seg["start"] * 1000)
            full_audio = full_audio.overlay(seg_sound, position=start_ms)

    os.makedirs("audio", exist_ok=True)
    dubbed_full_path = "audio/dubbed_full.wav"
    full_audio.export(dubbed_full_path, format="wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", dubbed_full_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        output_path
    ]
    subprocess.run(cmd, check=True)

    yield Event(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text=f"Final Assembly — Muxed dubbed audio with original video -> {output_path}"
                )
            ],
        )
    )
    yield Event(output=MuxResult(output_path=output_path))


# ---------------------------------------------------------------------------
# ADK 2.x Workflow — named FunctionNode instances (avoids duplicate-name error)
# ---------------------------------------------------------------------------

node_parse = FunctionNode(func=parse_input, name="parse_input")
node_extract = FunctionNode(
    func=extract_audio, name="extract_audio", retry_config=RetryConfig(max_attempts=2)
)
node_transcribe = FunctionNode(
    func=transcribe_audio, name="transcribe_audio", retry_config=RetryConfig(max_attempts=2)
)
node_translate = FunctionNode(
    func=translate_segments, name="translate_segments", retry_config=RetryConfig(max_attempts=3)
)
node_synthesise = FunctionNode(
    func=synthesise_segments, name="synthesise_segments", retry_config=RetryConfig(max_attempts=2)
)
node_mux = FunctionNode(func=mux_video, name="mux_video", retry_config=RetryConfig(max_attempts=2))

root_agent = Workflow(
    name="VideoLocalizerWorkflow",
    description=(
        "Localises video audio to Kannada (ಕನ್ನಡ) using a 5-stage local pipeline: "
        "ffmpeg → faster-whisper → Ollama (gemma2:2b) → MeloTTS (KN, CPU) → ffmpeg mux."
    ),
    edges=[
        ("START", node_parse),
        (node_parse, node_extract),
        (node_extract, node_transcribe),
        (node_transcribe, node_translate),
        (node_translate, node_synthesise),
        (node_synthesise, node_mux),
    ],
)
