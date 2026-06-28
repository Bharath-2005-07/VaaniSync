"""
TranslationAgent — Stage 3
Uses Ollama (gemma2:2b) via REST API to translate each transcription
segment into Kannada, preserving timestamps exactly.
"""

import json
from pathlib import Path

import requests
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma2:2b"

def _translate_batch(batch_segments: list[dict], model: str = DEFAULT_MODEL) -> list[dict]:
    """Translate a batch of segments together using local Ollama."""
    input_data = [{"id": i + 1, "text": seg["text"]} for i, seg in enumerate(batch_segments)]
    input_str = json.dumps(input_data, ensure_ascii=False, indent=2)

    prompt = (
        "You are an expert English-to-Kannada video dubbing translator. Your job is to translate subtitle segments accurately, maintaining the overall meaning, tone, and context of the video. \n\n"
        "CRITICAL RULES:\n"
        "1. Translate the meaning of the entire conversation naturally into spoken, conversational Kannada (ಕನ್ನಡ)—do not perform literal word-for-word translations.\n"
        "2. Because English is Subject-Verb-Object and Kannada is Subject-Object-Verb, adjust the sentence structure across adjacent segments so it sounds grammatically flawless to a native Kannada speaker. Ensure that the meaning in both languages is same.\n"
        "3. Keep the translation concise so it can be spoken within the allocated time windows.\n"
        "4. Return ONLY a valid JSON array matching the exact index structure provided. Do not include markdown formatting or conversational filler.\n\n"
        f"Input JSON to translate:\n{input_str}\n\n"
        "Expected Output JSON:"
    )

    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=60,
        )
        r.raise_for_status()
        val = r.json()["response"].strip()

        # Clean markdown fences
        if val.startswith("```"):
            lines = val.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            val = "\n".join(lines).strip()

        # Find first '[' and last ']'
        start_idx = val.find("[")
        end_idx = val.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            val = val[start_idx:end_idx + 1]

        parsed = json.loads(val)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
                    item["text"] = item["text"].strip()
            return parsed
    except Exception:
        pass

    # Return input data as fallback
    return input_data

def check_ollama_connection(model: str = DEFAULT_MODEL) -> dict:
    """Verify Ollama is running and the specified model is available."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            has_model = any(model in m or m in model for m in models)
            return {"running": True, "model": model, "available": has_model}
    except Exception:
        pass
    return {"running": False, "model": model, "available": False}

def translate_segments(segments_path: str, model: str = DEFAULT_MODEL) -> dict:
    """Translate segments from file into Kannada and write translated_segments.json."""
    segments_file = Path(segments_path)
    if not segments_file.exists():
        return {"success": False, "error": f"File not found: {segments_path}"}

    segments = json.loads(segments_file.read_text(encoding="utf-8"))
    translated = []
    errors = []

    try:
        translated_batch = _translate_batch(segments, model)
        translation_map = {}
        for item in translated_batch:
            if isinstance(item, dict) and "id" in item and "text" in item:
                try:
                    translation_map[int(item["id"])] = item["text"]
                except Exception:
                    pass

        for i, seg in enumerate(segments):
            kannada = translation_map.get(i + 1, seg["text"])
            translated.append({
                "start": seg["start"],
                "end": seg["end"],
                "original_text": seg["text"],
                "text": kannada,
            })
    except Exception as e:
        errors.append(str(e))
        for seg in segments:
            translated.append({
                "start": seg["start"],
                "end": seg["end"],
                "original_text": seg["text"],
                "text": seg["text"],
            })

    output_dir = Path("transcripts")
    output_dir.mkdir(exist_ok=True)
    output_json = output_dir / "translated_segments.json"
    output_json.write_text(json.dumps(translated, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "success": True,
        "translated_path": str(output_json),
        "segment_count": len(translated),
        "translation_errors": errors,
        "error": None,
    }

# ADK Agent definition
translation_agent = Agent(
    name="TranslationAgent",
    model="gemini-2.0-flash",
    description=(
        "Translates transcription segments to Kannada (ಕನ್ನಡ) using Ollama (gemma2:2b) locally. "
        "Preserves all {start, end} timestamps exactly. Outputs translated_segments.json."
    ),
    instruction=(
        "You are the TranslationAgent. First call check_ollama_connection to verify Ollama is running. "
        "If it is not running, instruct the user to run 'ollama serve' and 'ollama pull gemma2:2b', then stop. "
        "Otherwise call translate_segments with the segments_path from the orchestrator. "
        "Report the translated_path, segment_count, and any translation_errors back."
    ),
    tools=[FunctionTool(check_ollama_connection), FunctionTool(translate_segments)],
)
