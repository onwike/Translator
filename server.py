"""FastAPI server: transcribe Twi/Igbo audio and translate to English.

Pipeline:
  audio bytes (webm/ogg/wav) -> 16kHz mono float32 (ffmpeg)
                              -> MMS speech recognition (Twi or Igbo)
                              -> NLLB-200 translation to English
"""
from __future__ import annotations

import io
import logging
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    Wav2Vec2ForCTC,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("translator")

LangCode = Literal["twi", "ibo"]

# MMS uses ISO 639-3 codes for its target language adapter.
MMS_LANG = {"twi": "twi", "ibo": "ibo"}

# NLLB-200 uses BCP-47-ish FLORES codes.
NLLB_SRC = {"twi": "twi_Latn", "ibo": "ibo_Latn"}
NLLB_TGT = "eng_Latn"

MMS_MODEL_ID = "facebook/mms-1b-all"
NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


@lru_cache(maxsize=1)
def _mms():
    log.info("Loading MMS model %s on %s", MMS_MODEL_ID, DEVICE)
    processor = AutoProcessor.from_pretrained(MMS_MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MMS_MODEL_ID).to(DEVICE)
    model.eval()
    return processor, model


@lru_cache(maxsize=1)
def _nllb():
    log.info("Loading NLLB model %s on %s", NLLB_MODEL_ID, DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL_ID, torch_dtype=DTYPE
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


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
    # MMS exposes a per-language adapter that swaps both tokenizer and CTC head.
    processor.tokenizer.set_target_lang(target)
    model.load_adapter(target)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0].strip()


def translate_nllb(text: str, lang: LangCode) -> str:
    if not text:
        return ""
    tokenizer, model = _nllb()
    tokenizer.src_lang = NLLB_SRC[lang]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    forced_bos = tokenizer.convert_tokens_to_ids(NLLB_TGT)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_new_tokens=256,
            num_beams=2,
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


app = FastAPI(title="Twi/Igbo → English live translator")

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def health():
    return {"status": "ok", "device": DEVICE}


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
    translation = translate_nllb(transcript, lang)
    log.info("[%s] %r -> %r", lang, transcript, translation)
    return {"transcript": transcript, "translation": translation}


@app.on_event("startup")
def warmup():
    # Pre-load both models so the first request isn't a 30s cold-start.
    _mms()
    _nllb()
    log.info("Models loaded; ready.")
