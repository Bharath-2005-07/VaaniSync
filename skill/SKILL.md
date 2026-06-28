---
name: video-localizer
description: Automates video translation and audio localization on a local machine. Converts video speech from the original language into Kannada, matching timestamps and synthesising natural speech locally using MeloTTS on CPU.
target_language: Kannada
tts_engine: MeloTTS
llm_backend: Ollama
---

# Video Localizer & Audio Dubbing Agent (Kannada)

## Objective
Your goal is to build, orchestrate, and execute a local multi-step AI video dubbing pipeline. You will coordinate the tools required to extract, transcribe, translate, synthesize, and recombine video assets entirely on the user's local machine.

## Technical Architecture Overview
You must implement a 5-stage pipeline using Python, keeping execution isolated to local processing:

1. **Audio Extraction** — Use `ffmpeg` to rip audio streams from the input video.
2. **Transcription (Speech-to-Text)** — Use `faster-whisper` locally for exact text with precise word/segment-level timestamps.
3. **Translation (LLM)** — Use a local LLM via `Ollama` (recommended: `gemma2:2b` or `mistral:7b-instruct-q4_K_M`) to translate text segment-by-segment to **Kannada**, preserving structural boundaries.
4. **Voice Synthesis (TTS)** — Use `MeloTTS` (CPU-optimised, no GPU required) to synthesise Kannada speech for each translated segment, with duration-matching via `pydub`.
5. **Muxing** — Use `ffmpeg` or `MoviePy` to recombine the video stream with the new Kannada audio track.

---

## Pipeline Stages & Tool Responsibilities

### Stage 1 — Audio Extraction
- **Tool:** `ffmpeg`
- **Input:** Original video file (e.g., `video/video1.mp4`)
- **Output:** Raw audio file (`audio/original_audio.wav`, 16kHz mono preferred for Whisper)
- **Command pattern:**
  ```bash
  ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio/original_audio.wav
  ```

### Stage 2 — Transcription
- **Tool:** `faster-whisper` (local, CPU or GPU)
- **Input:** `audio/original_audio.wav`
- **Output:** `transcripts/segments.json` — list of `{start, end, text}` segment objects
- **Key requirement:** Preserve exact timestamps for synchronisation in later stages.
- **Model sizes:** Use `medium` for balance; `large-v3` for best accuracy.

### Stage 3 — Translation
- **Tool:** `Ollama` with `gemma2:2b` (or `mistral:7b-instruct-q4_K_M`)
- **Input:** `transcripts/segments.json`
- **Output:** `transcripts/translated_segments.json`
- **Core Translation Strategy (Context-Aware Batching):** Do NOT translate segments one by one in absolute isolation. Instead, the agent must pass batches of sentences (or the full block of text with indices) to the LLM so it understands the global context and narrative flow of the video. The LLM must reconstruct the meaning naturally in Kannada and output it mapped back to the corresponding segment indices.

- **System Prompt Pattern:**
  ```text
  You are an expert English-to-Kannada video dubbing translator. Your job is to translate subtitle segments accurately, maintaining the overall meaning, tone, and context of the video. 

  CRITICAL RULES:
  1. Translate the meaning of the entire conversation naturally into spoken, conversational Kannada (ಕನ್ನಡ)—do not perform literal word-for-word translations.
  2. Because English is Subject-Verb-Object and Kannada is Subject-Object-Verb, adjust the sentence structure across adjacent segments so it sounds grammatically flawless to a native Kannada speaker.
  3. Keep the translation concise so it can be spoken within the allocated time windows.
  4. Return ONLY a valid JSON array matching the exact index structure provided. Do not include markdown formatting or conversational filler.

  Input JSON to translate:
  [
    {"id": 1, "text": "Welcome back to the channel. Today we are going to"},
    {"id": 2, "text": "learn how a blockchain works from scratch."}
  ]

  Expected Output JSON:
  [
    {"id": 1, "text": "ಚಾನೆಲ್ಗೆ ಮರಳಿ ಸ್ವಾಗತ. ಇಂದು ನಾವು"},
    {"id": 2, "text": "ಬ್ಲಾಕ್ಚೈನ್ ಮೊದಲಿನಿಂದ ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ ಎಂದು ಕಲಿಯಲಿದ್ದೇವೆ."}
  ]
  ```
- **Model pull command:** `ollama pull gemma2:2b`

### Stage 4 — Voice Synthesis (TTS)
- **Tool:** `MeloTTS` (CPU-optimised, lightweight, no GPU required)
- **Why MeloTTS?** Runs entirely on CPU, low memory footprint, supports Indic language phoneme sets, fast inference per segment.
- **Input:** `transcripts/translated_segments.json`
- **Output:** `audio/dubbed_segments/segment_NNN.wav` per segment
- **Key requirements:**
  - Synthesise Kannada speech for each translated text segment.
  - Use `pydub` to trim or pad each segment WAV to fit exactly within `[start, end]` duration window.
  - If segment audio is shorter than window → append silence padding.
  - If segment audio is longer than window → apply `pydub` speed-up (1.1x–1.3x) before padding.
- **Language code:** `KN` (Kannada) for MeloTTS
- **Sample usage:**
  ```python
  from melo.api import TTS
  model = TTS(language="KN", device="cpu")
  speaker_ids = model.hps.data.spk2id
  model.tts_to_file(text, speaker_ids["KN"], output_path, speed=1.0)
  ```

### Stage 5 — Muxing (Final Assembly)
- **Tool:** `ffmpeg` or `MoviePy`
- **Input:** Original video (video stream only) + all `audio/dubbed_segments/*.wav`
- **Output:** `output/localized_video.mp4`
- **Steps:**
  1. Concatenate all dubbed audio segments into a single WAV: `audio/dubbed_full.wav`
  2. Strip original audio from video.
  3. Merge video stream with `dubbed_full.wav`.
  ```bash
  ffmpeg -i input.mp4 -i audio/dubbed_full.wav -c:v copy -map 0:v:0 -map 1:a:0 output/localized_video.mp4
  ```

---

## Directory Structure
```
lang-to-lang/
├── skill/
│   └── SKILL.md              ← This file
├── video/
│   └── video1.mp4            ← Input video
├── audio/
│   ├── original_audio.wav    ← Extracted audio
│   ├── dubbed_segments/      ← Per-segment TTS output
│   └── dubbed_full.wav       ← Final assembled audio
├── transcripts/
│   ├── segments.json         ← Whisper output
│   └── translated_segments.json ← LLM-translated output
├── output/
│   └── localized_video.mp4   ← Final dubbed video
├── video_localizer/
│   └── agent.py              ← ADK agent entry point
├── .venv/                    ← Local Python venv (google-adk installed)
└── agents-cli-manifest.yaml  ← ADK project manifest
```

---

## Agent Responsibilities (ADK Multi-Agent Design)

This skill uses a **sequential multi-agent** pattern via Google ADK:

| Agent | Role |
|---|---|
| `OrchestratorAgent` | Root agent; receives user request, resolves file paths, dispatches sub-agents in order |
| `ExtractionAgent` | Runs ffmpeg to extract audio; returns path to WAV |
| `TranscriptionAgent` | Runs faster-whisper; returns `segments.json` |
| `TranslationAgent` | Calls local Ollama LLM; returns `translated_segments.json` |
| `SynthesisAgent` | Runs XTTS v2 / MeloTTS per segment; returns per-segment WAVs |
| `MuxingAgent` | Assembles final video with ffmpeg/MoviePy |

---

## Prerequisites

### Required Local Tools
| Tool | Install | Notes |
|---|---|---|
| `ffmpeg` | `winget install ffmpeg` | Audio/video processing |
| `faster-whisper` | `.venv\Scripts\pip install faster-whisper` | Use `small` model on CPU |
| `Ollama` | https://ollama.com/download | Then: `ollama pull gemma2:2b` |
| `MeloTTS` | `.venv\Scripts\pip install melo-tts` | CPU-native TTS, Kannada support |
| `pydub` | `.venv\Scripts\pip install pydub` | Audio segment trimming/padding |
| `moviepy` | `.venv\Scripts\pip install moviepy` | Final video assembly |
| `google-adk` | Already installed in `.venv` | ADK 2.3.0 |

### CPU Laptop Tips
- Use `faster-whisper` with model `small` or `base` for acceptable speed on CPU.
- Use `gemma2:2b` with Ollama — it's quantized (2B params) and runs comfortably on 8GB RAM.
- MeloTTS is already CPU-native — no special config needed.
- Process video segments in batches to avoid memory pressure.

---

## Invocation Pattern

When the user says something like:
- *"Dub this video into Spanish"*
- *"Convert audio to French and keep my voice"*
- *"Localize video1.mp4 to Tamil"*

You MUST:
1. Confirm the **input video path** and **target language**.
2. Check that all prerequisite tools are installed.
3. Execute the 5-stage pipeline in order.
4. Report the output path to the user on completion.

---

## Error Handling Guidelines
- If `ffmpeg` is not found → prompt user: `winget install ffmpeg`, then restart terminal.
- If Whisper model download fails → use `base` model: `WhisperModel("base", device="cpu")`.
- If Ollama is not running → instruct user: `ollama serve` then `ollama pull gemma2:2b`.
- If MeloTTS Kannada synthesis fails → verify language code is `"KN"` and `melo-tts` is installed in venv.
- If segment audio is too long for the window → speed up with `pydub`: `audio.speedup(playback_speed=1.2)`.
- If Ollama returns non-Kannada output → add explicit instruction: `"Respond ONLY in Kannada script (ಕನ್ನಡ). No English."`

---

## Notes
- All processing is **100% local** — no cloud API calls for the core pipeline.
- **Target language:** Kannada (ಕನ್ನಡ) — language code `KN`
- **TTS engine:** MeloTTS — CPU-native, no GPU required
- **LLM backend:** Ollama with `gemma2:2b` — best accuracy/speed for CPU laptops
- **Whisper model:** `small` recommended for CPU; `base` for faster but less accurate transcription
- The `google-adk` framework handles agent orchestration, tool calling, and session state.
- ADK version in use: **2.3.0**
- This skill is designed to work with the `adk web` dev UI for interactive testing.
- Workflow reference: see `information/workflow_graph.md` for the full pipeline diagram.
