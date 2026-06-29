---
name: video-localizer
description: Automates video translation and audio localization on a local machine. Converts video speech from the original language into any dynamically requested language (like Spanish, French, Kannada, Hindi, Telugu, etc.), matching timestamps and synthesising natural speech locally using MeloTTS/edge-tts on CPU.
target_language: Dynamic (Defaults to Kannada)
tts_engine: MeloTTS / Edge-TTS
llm_backend: Ollama
---

# Video Localizer & Audio Dubbing Agent (Kannada)

## Objective
Your goal is to build, orchestrate, and execute a local multi-step AI video dubbing pipeline. You will coordinate the tools required to extract, transcribe, translate, synthesize, and recombine video assets entirely on the user's local machine.

## Technical Architecture Overview
You must implement a 5-stage pipeline using Python, keeping execution isolated to local processing:

1. **Audio Extraction** — Use `ffmpeg` to rip audio streams from the input video and denoise via `afftdn`.
2. **Transcription (Speech-to-Text)** — Use `faster-whisper` locally for exact text with precise segment-level timestamps.
3. **Translation (LLM)** — Use the `translation_agent` (backed by a local LLM via `Ollama` running `gemma2:2b` or online fallback) to translate text segment-by-segment, preserving structural boundaries.
4. **Voice Synthesis (TTS)** — Use `MeloTTS` (CPU-optimised, language='KN') or `edge-tts` (as fallback) to synthesise Kannada speech for each translated segment, with voice cloning via `OpenVoice V2` and duration-matching via `pydub` (speed-up/padding).
5. **Muxing** — Use `ffmpeg` to recombine the video stream with the new dubbed audio track.

---

## Pipeline Stages & Agent Responsibilities

This system uses a **sequential multi-agent** pattern managed via the `VideoLocalizerWorkflow` class (a Google ADK 2.0 Workflow).

### 📋 Stage 0 — Input Parser Agent (`parse_input` Node)
* **Responsibility**: Process initial request query to extract pipeline arguments.
* **Input**: User prompt (e.g. *"Dub video/virat_kohli.mp4 into Kannada"*) or `PipelineInput` schema.
* **Output**: Validated `PipelineInput` containing target language, voice gender preference, and resolved video file path.
* **Heuristics**: Scans the `video/` folder for matching filenames and automatically resolves paths.

### 🔊 Stage 1 — Audio Extraction Agent (`ExtractionAgent` / `extract_audio` Node)
- **Tool:** `ffmpeg`
- **Input:** Original video file (e.g., `video/virat_kohli.mp4`)
- **Output:** Denoised raw audio file (`audio/original_audio.wav`, 16kHz mono PCM)
- **Command pattern:**
  ```bash
  ffmpeg -y -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 -af afftdn audio/original_audio.wav
  ```

### 📝 Stage 2 — Speech Transcription Agent (`TranscriptionAgent` / `transcribe_audio` Node)
- **Tool:** `faster-whisper` (local, CPU quantized to `int8`)
- **Input:** `audio/original_audio.wav`
- **Output:** `transcripts/segments.json` — list of `{start, end, text}` segment objects
- **Key requirement:** Disables hallucination-prone configurations by setting `word_timestamps=False` and `condition_on_previous_text=False`.

### 🌐 Stage 3 — Translation Agent (`TranslationAgent` / `translate_segments` Node)
- **Tool:** ADK `translation_agent` (in `video_localizer/agents/translation.py`) calling local Ollama (`gemma2:2b`) or fallback.
- **Input:** `transcripts/segments.json`
- **Output:** `transcripts/translated_segments.json`
- **Core Translation Strategy (Context-Aware Batching):** Multiple subtitle segments are joined with ` ||| ` and translated in a single payload to capture global context. If it fails, falls back to translating segments individually.

### 🗣️ Stage 4 — Neural Voice Synthesis Agent (`SynthesisAgent` / `synthesise_segments` Node)
- **Tool:** `MeloTTS` (CPU-optimised, language='KN') or `edge-tts` fallback + `OpenVoice V2` for voice cloning.
- **Input:** `transcripts/translated_segments.json`
- **Output:** Timing-aligned WAV files under `audio/dubbed_segments/segment_NNN.wav`
- **Voice Cloning Strategy**: When `checkpoints_v2` is present, extracts a tone-color embedding from the original speaker (`audio/original_audio.wav`) and morphs the synthesized audio to clone the original voice offline.
- **Concurrent Processing**: Concurrently synthesizes segments using `ThreadPoolExecutor` (max 4 workers) with a threading Lock (`melo_lock`) to serialize PyTorch execution and prevent crashes.
- **Pacing Control**: Appends silence using `pydub` if a segment is short, or speeds it up (up to 2.0x) using FFmpeg `atempo` filter if it is long.

### 🎬 Stage 5 — Assembly & Muxing Agent (`MuxingAgent` / `mux_video` Node)
- **Tool:** `pydub` + `ffmpeg`
- **Input:** Original video + `audio/dubbed_segments/*.wav`
- **Output:** `output/localized_video.mp4` (e.g., `output/virat_kohli.mp4`)
- **Steps:** Overlays dubbed segments onto a silent canvas of the original length, saves to `audio/dubbed_full.wav`, and multiplexes:
  ```bash
  ffmpeg -y -i input.mp4 -i audio/dubbed_full.wav -c:v copy -map 0:v:0 -map 1:a:0 output/localized_video.mp4
  ```

---

## Directory Structure
```text
lang-to-lang/
├── .agents/
│   └── skills/
│       └── video-localizer/
│           └── SKILL.md          # Custom agent skill definition file
├── .env                          # Local environment variables (optional keys)
├── .venv/                        # Local Python virtual environment
├── audio/                        # Temporary processing directory for audio
│   ├── original_audio.wav        # Stage 1: Extracted and denoised original audio
│   ├── dubbed_segments/          # Stage 4: Concurrent segment TTS outputs
│   └── dubbed_full.wav           # Stage 5: Assembled dubbed audio track
├── checkpoints_v2/               # OpenVoice V2 converter model weights folder
│   └── converter/
│       ├── checkpoint.pth        # Converter PyTorch weights
│       └── config.json           # Converter configuration parameters
├── information/                  # Project documentation assets
│   ├── pipeline_run.png          # Web UI execution screenshot
│   ├── workflow_graph.md         # Pipeline flowchart and detailed architecture
│   └── architecture_workflow_diagram.png # High-resolution architecture visual
├── inputs/                       # User-supplied media input files
├── output/                       # Final dubbed video output files
│   ├── video2.mp4                # Stage 5: Dubbed output for video2
│   ├── video5.mp4                # Stage 5: Dubbed output for video5
│   └── virat_kohli.mp4           # Stage 5: Dubbed output for Virat Kohli
├── processed/                    # Speaker embedding cache (cleaned post-run)
├── pyproject.toml                # Build configuration and dependency specifications
├── requirements.txt              # Primary project pip packages list
├── run_dubbing.bat               # Interactive drag-and-drop batch script
├── run_guide.md                  # Quick run commands cheat sheet
├── skill/
│   └── SKILL.md                  # Reusable skill documentation
├── tests/                        # Automated unit and integration tests
│   ├── test_pipeline.py          # Pytest suite with mocked services
│   └── eval/                     # Evaluation configurations and datasets
│       ├── eval_config.yaml
│       └── eval_dataset.json
├── transcripts/                  # Temporary translation segments storage
│   ├── segments.json             # Stage 2: Whisper speech timestamps & text
│   └── translated_segments.json  # Stage 3: Kannada translation with metadata
├── video/                        # Input video files directory
│   ├── video3.mp4                # Secondary testing video input
│   └── virat_kohli.mp4           # Primary reference video input
├── video_localizer/              # Main agent workflow package
│   ├── __init__.py               # Exports discovery root agent workflow
│   ├── agent.py                  # Orchestrator & FunctionNode stage handlers
│   └── agents/                   # Sub-agent modules (e.g., translation)
│       ├── __init__.py
│       └── translation.py
├── agents-cli-manifest.yaml      # ADK project registration manifest
├── working.md                    # In-depth technical breakdown of workflow
└── README.md                     # Project homepage GitHub README
```

---

## Agent Responsibilities (ADK Multi-Agent Design)

This skill uses a sequential multi-agent pattern orchestrated via Google ADK:

| Agent / Node | Role |
|---|---|
| `VideoLocalizerWorkflow` | Root orchestrator workflow; manages shared state in `Context.state` and drives execution |
| `parse_input` | Parsing node; validates arguments and resolves local paths |
| `ExtractionAgent` | Audio node; runs ffmpeg to extract and denoise wav stream |
| `TranscriptionAgent` | Transcription node; runs faster-whisper on CPU to generate segments |
| `TranslationAgent` | Translation agent; calls local Ollama LLM / online translator in batches |
| `SynthesisAgent` | TTS node; generates MeloTTS/Edge-TTS segments and runs OpenVoice V2 cloning |
| `MuxingAgent` | Mux node; overlays audio onto timeline and merges with video container |

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
| `google-adk` | Already installed in `.venv` | ADK 2.3.0 |

---

## Invocation Pattern

When the user requests:
- *"Dub this video into Spanish"*
- *"Convert audio to French and keep my voice"*
- *"Localize video/virat_kohli.mp4 to Kannada"*

You MUST:
1. Confirm the **input video path** and **target language**.
2. Check that all prerequisite tools are installed and Ollama is serving (if offline).
3. Execute the workflow pipeline nodes sequentially.
4. Report the resulting dubbed video output path (located in `output/`).
