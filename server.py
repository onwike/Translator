"""FastAPI server: live Twi/Igbo speech -> English translation.

Two modes:

1. On-device streaming (default, WS /ws/transcribe):
   The browser streams 500 ms PCM16 mono frames over a WebSocket. The server
   keeps a rolling buffer (capped at 30 s) per connection and continuously
   re-transcribes the whole window with MMS, then translates with a Grok text
   model when XAI_API_KEY is set (NLLB-200 fallback otherwise). Partial
   results revise in place; a segment finalizes on trailing silence, on the
   30 s cap, or on an explicit client flush. The buffer doubles as the error
   guard: a failed pass just retries on the next pass over the same audio.

2. Grok Live (experimental, POST /grok-live/token):
   The browser connects directly to xAI's realtime voice API with an
   ephemeral token minted here, with Grok instructed to act as a pure
   simultaneous interpreter. Igbo and Twi are NOT officially supported by
   grok-voice; this mode exists to test empirically whether it understands
   them.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    Wav2Vec2ForCTC,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("translator")


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines, # comments."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(Path(__file__).parent / ".env")

LangCode = Literal["twi", "ibo"]

# MMS uses ISO 639-3 codes for its target language adapter.
MMS_LANG = {"twi": "twi", "ibo": "ibo"}

# NLLB-200 uses BCP-47-ish FLORES codes.
NLLB_SRC = {"twi": "twi_Latn", "ibo": "ibo_Latn"}
NLLB_TGT = "eng_Latn"

LANG_NAME = {"twi": "Twi (Akan)", "ibo": "Igbo"}

# BCP-47 hints for Grok's input transcription (best effort; neither language
# is on Grok's official list).
BCP47_HINT = {"twi": "tw", "ibo": "ig"}

MMS_MODEL_ID = "facebook/mms-1b-all"
NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"

XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_BASE = "https://api.x.ai/v1"
GROK_TEXT_MODEL = os.environ.get("GROK_TEXT_MODEL", "grok-4.3")
GROK_VOICE_MODEL = os.environ.get("GROK_VOICE_MODEL", "grok-voice-latest")

SAMPLE_RATE = 16000
MAX_BUFFER_SECONDS = 30.0   # rolling window cap; forces finalization
MIN_AUDIO_SECONDS = 0.3     # skip windows shorter than this
SILENCE_SECONDS = 0.7       # trailing quiet needed to finalize a segment
SILENCE_RMS = 0.010         # RMS below this counts as silence
HISTORY_LINES = 5           # finalized English lines kept as Grok context

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


@lru_cache(maxsize=1)
def _xai_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=XAI_BASE,
        headers={"Authorization": f"Bearer {XAI_API_KEY}"},
        timeout=10.0,
    )


# The MMS/NLLB models hold mutable state (language adapter, src_lang) and are
# shared across connections, so inference is serialized.
_mms_lock = threading.Lock()
_mms_adapter: str | None = None
_nllb_lock = threading.Lock()


def transcribe_mms(audio: np.ndarray, lang: LangCode) -> str:
    global _mms_adapter
    processor, model = _mms()
    with _mms_lock:
        target = MMS_LANG[lang]
        # MMS exposes a per-language adapter that swaps both tokenizer and CTC head.
        if _mms_adapter != target:
            processor.tokenizer.set_target_lang(target)
            model.load_adapter(target)
            _mms_adapter = target
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(pred_ids)[0].strip()


def translate_nllb(text: str, lang: LangCode) -> str:
    if not text:
        return ""
    tokenizer, model = _nllb()
    with _nllb_lock:
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


async def translate_grok(text: str, lang: LangCode, history: list[str]) -> str | None:
    """Translate via Grok's chat API. Returns None on any failure so the
    caller can fall back to NLLB."""
    if not text or not XAI_API_KEY:
        return None
    system = (
        f"You are a translator. The user message is a possibly noisy ASR "
        f"transcript of spoken {LANG_NAME[lang]}. Translate it into natural "
        f"English. Output only the translation, nothing else."
    )
    messages = [{"role": "system", "content": system}]
    if history:
        messages.append({
            "role": "system",
            "content": "Preceding translations, for context:\n" + "\n".join(history),
        })
    messages.append({"role": "user", "content": text})
    try:
        resp = await _xai_client().post("/chat/completions", json={
            "model": GROK_TEXT_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 300,
        })
        resp.raise_for_status()
        out = resp.json()["choices"][0]["message"]["content"].strip()
        return out or None
    except Exception as exc:
        log.warning("Grok translation failed, falling back to NLLB: %s", exc)
        return None


def interpreter_instructions(lang: LangCode) -> str:
    return (
        f"You are a simultaneous interpreter. The user speaks {LANG_NAME[lang]}. "
        "Never converse, never answer questions, never add commentary. "
        "For every utterance you hear, output only its English translation. "
        "If the audio is unintelligible, output '[unclear]'."
    )


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _trailing_silence(samples: np.ndarray) -> bool:
    tail_len = int(SILENCE_SECONDS * SAMPLE_RATE)
    if samples.size < tail_len:
        return False
    return _rms(samples[-tail_len:]) < SILENCE_RMS


class _Stream:
    """Per-connection state shared between the receiver and the worker."""

    def __init__(self) -> None:
        self.buf = np.zeros(0, dtype=np.float32)
        self.new_data = asyncio.Event()
        self.flush = False
        self.closed = False


async def _receive_frames(ws: WebSocket, st: _Stream) -> None:
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                samples = np.frombuffer(msg["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                st.buf = np.concatenate([st.buf, samples])
            elif msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except ValueError:
                    continue
                if data.get("type") == "flush":
                    st.flush = True
            st.new_data.set()
    finally:
        st.closed = True
        st.new_data.set()


async def _transcribe_worker(ws: WebSocket, st: _Stream, lang: LangCode) -> None:
    history: list[str] = []
    last_len = -1
    while not st.closed:
        if len(st.buf) == last_len and not st.flush:
            st.new_data.clear()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(st.new_data.wait(), timeout=1.0)
            continue

        snapshot = st.buf  # receiver replaces st.buf, never mutates it in place
        last_len = len(snapshot)
        flushing = st.flush
        duration = len(snapshot) / SAMPLE_RATE
        finalize = flushing or duration >= MAX_BUFFER_SECONDS or _trailing_silence(snapshot)

        if duration < MIN_AUDIO_SECONDS or _rms(snapshot) < SILENCE_RMS:
            # Nothing worth transcribing (too short, or pure silence while
            # the mic idles). Drop the window once it would have finalized so
            # idle silence never accumulates toward the 30 s cap.
            if finalize:
                st.buf = st.buf[len(snapshot):]
                last_len = -1
                if flushing:
                    break
            continue

        # A failure below leaves the buffer untouched, so the same (grown)
        # window is retried on the next pass — the retranslation guard.
        try:
            transcript = await asyncio.to_thread(transcribe_mms, snapshot, lang)
            translation, engine = "", "nllb"
            if transcript:
                grok = await translate_grok(transcript, lang, history)
                if grok is not None:
                    translation, engine = grok, "grok"
                else:
                    translation = await asyncio.to_thread(translate_nllb, transcript, lang)
        except Exception as exc:
            log.exception("transcription pass failed; will retry: %s", exc)
            await asyncio.sleep(0.5)
            last_len = -1
            continue

        if st.closed:
            break
        try:
            await ws.send_json({
                "partial": not finalize,
                "transcript": transcript,
                "translation": translation,
                "engine": engine,
            })
        except Exception:
            break

        if finalize:
            log.info("[%s/%s] %r -> %r", lang, engine, transcript, translation)
            if translation:
                history.append(translation)
                del history[:-HISTORY_LINES]
            st.buf = st.buf[len(snapshot):]
            last_len = -1
            if flushing:
                break


app = FastAPI(title="Twi/Igbo → English live translator")

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "grok": bool(XAI_API_KEY),
        "grok_text_model": GROK_TEXT_MODEL,
        "grok_voice_model": GROK_VOICE_MODEL,
    }


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    lang = ws.query_params.get("lang", "")
    if lang not in MMS_LANG:
        await ws.close(code=4400, reason="lang must be 'twi' or 'ibo'")
        return
    await ws.accept()
    st = _Stream()
    receiver = asyncio.create_task(_receive_frames(ws, st))
    try:
        await _transcribe_worker(ws, st, lang)  # type: ignore[arg-type]
    finally:
        st.closed = True
        receiver.cancel()
        with contextlib.suppress(Exception):
            await ws.close()


class TokenRequest(BaseModel):
    lang: LangCode = "twi"


@app.post("/grok-live/token")
async def grok_live_token(req: TokenRequest):
    """Mint an ephemeral xAI token so the browser can open the realtime
    WebSocket itself without ever seeing the API key."""
    if not XAI_API_KEY:
        raise HTTPException(404, "XAI_API_KEY not configured on the server")
    try:
        resp = await _xai_client().post(
            "/realtime/client_secrets", json={"expires_after": {"seconds": 300}}
        )
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"could not reach xAI: {exc}") from exc
    if resp.status_code >= 400:
        raise HTTPException(502, f"xAI token request failed: {resp.text[:300]}")
    data = resp.json()
    token = (
        data.get("value")
        or (data.get("client_secret") or {}).get("value")
        or data.get("token")
    )
    if not token:
        raise HTTPException(502, "unexpected token response from xAI")
    # The browser passes the token as WebSocket subprotocol "xai-client-secret.<token>".
    prefix = "xai-client-secret."
    if token.startswith(prefix):
        token = token[len(prefix):]
    return {
        "token": token,
        "model": GROK_VOICE_MODEL,
        "session": {
            "instructions": interpreter_instructions(req.lang),
            "output_modalities": ["text"],
            "turn_detection": {"type": "server_vad"},
            "reasoning": {"effort": "none"},
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "transcription": {
                        "model": "grok-transcribe",
                        "language_hint": BCP47_HINT[req.lang],
                    },
                },
            },
        },
    }


@app.on_event("startup")
def warmup():
    # Pre-load both models so the first connection isn't a 30s cold-start.
    _mms()
    _nllb()
    log.info("Models loaded; ready. Grok configured: %s", bool(XAI_API_KEY))
