"""FastAPI server: transcribe Twi/Igbo audio and translate to English with Claude.

Pipeline:
  audio bytes (webm/ogg/wav) -> 16kHz mono float32 (ffmpeg)
                              -> MMS speech recognition (Twi or Igbo)
                              -> Claude Haiku 4.5 translation to English
"""
from __future__ import annotations

import logging
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Literal

import anthropic
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoProcessor, Wav2Vec2ForCTC

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("translator")

LangCode = Literal["twi", "ibo"]

MMS_LANG = {"twi": "twi", "ibo": "ibo"}
LANG_NAME = {"twi": "Twi (Akan)", "ibo": "Igbo"}

MMS_MODEL_ID = "facebook/mms-1b-all"
CLAUDE_MODEL = "claude-haiku-4-5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's message from {language} "
    "into natural, fluent English. Output ONLY the English translation — no preamble, "
    "no explanations, no quotes, no commentary. Preserve meaning and tone faithfully. "
    "If the input is empty, fragmentary, or unintelligible, respond with an empty string."
)


@lru_cache(maxsize=1)
def _mms():
    log.info("Loading MMS model %s on %s", MMS_MODEL_ID, DEVICE)
    processor = AutoProcessor.from_pretrained(MMS_MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MMS_MODEL_ID).to(DEVICE)
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def _claude() -> anthropic.Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before starting the server: "
            "export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic()


def decode_audio(raw: bytes) -> np.ndarray:
    """Decode any browser-recorded container to 16kHz mono float32 via ffmpeg."""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "f32le", "-ac", "1", "-ar", "16000",
            "pipe:1",
        ],
        input=raw, capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise HTTPException(400, f"ffmpeg failed: {proc.stderr.decode(errors='ignore')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def transcribe_mms(audio: np.ndarray, lang: LangCode) -> str:
    processor, model = _mms()
    target = MMS_LANG[lang]
    processor.tokenizer.set_target_lang(target)
    model.load_adapter(target)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0].strip()


def translate_with_claude(text: str, lang: LangCode) -> str:
    if not text:
        return ""
    client = _claude()
    try:
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT.format(language=LANG_NAME[lang]),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": text}],
        )
    except anthropic.APIStatusError as e:
        log.warning("Claude API error: %s", e)
        raise HTTPException(502, f"Claude translation failed: {e.message}") from e

    return next(
        (b.text.strip() for b in response.content if b.type == "text"),
        "",
    )


app = FastAPI(title="Twi/Igbo → English live translator")

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def health():
    return {"status": "ok", "device": DEVICE, "claude_model": CLAUDE_MODEL}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    lang: LangCode = Form(...),
):
    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "empty audio")
    samples = decode_audio(raw)
    if samples.size < 16000 * 0.3:  # < 0.3s of audio
        return {"transcript": "", "translation": ""}
    transcript = transcribe_mms(samples, lang)
    translation = translate_with_claude(transcript, lang)
    log.info("[%s] %r -> %r", lang, transcript, translation)
    return {"transcript": transcript, "translation": translation}


@app.on_event("startup")
def warmup():
    _claude()  # Fail fast if ANTHROPIC_API_KEY is missing.
    _mms()
    log.info("MMS loaded, Claude client ready (%s).", CLAUDE_MODEL)
