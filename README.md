# Twi / Igbo → English live speech translator

A small web app that captures live microphone audio in Twi or Igbo, transcribes
it with **Meta MMS** (Massively Multilingual Speech, 1000+ languages), and
translates the transcript to English with **NLLB-200**.

```
mic → MediaRecorder (5s chunks) → FastAPI /transcribe
                                  → ffmpeg decode (16kHz mono)
                                  → MMS speech recognition
                                  → NLLB-200 translation
                                  → JSON {transcript, translation}
```

Browsers don't natively support Twi or Igbo speech recognition (and OpenAI
Whisper doesn't list them either), so the heavy lifting happens on the
server.

## Requirements

- Python 3.10+
- `ffmpeg` on PATH (used to decode the browser's webm/ogg/mp4 audio chunks)
- ~3 GB of disk for the models (downloaded on first run)
- A GPU is nice but not required; CPU works, just slower per chunk

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional: keep the model cache inside the project
export HF_HOME=$(pwd)/hf_cache
```

## Run

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>, pick **Twi** or **Igbo**, click **Start**, and
talk. The left pane shows the original transcript, the right pane shows the
English translation. The first request triggers a one-time model download
(~2.5 GB) and warmup.

## Notes & honest limits

- **Low-resource languages.** Twi and Igbo have far less training data than
  English/Spanish/Mandarin. Expect transcription errors, especially with
  background noise, multiple speakers, or heavy code-switching.
- **Near-live, not streaming.** Audio is sent in self-contained 3–8 second
  chunks (the page restarts the `MediaRecorder` on each interval so each blob
  has full container headers — required for ffmpeg to decode it). True
  word-by-word streaming would need a different model architecture (e.g.
  RNN-T) and a WebSocket protocol.
- **Models used**
  - Speech: [`facebook/mms-1b-all`](https://huggingface.co/facebook/mms-1b-all)
  - Translation: [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)
- **Privacy.** Audio leaves the browser to reach your own server; nothing goes
  to a third-party API.

## Adding more languages

MMS supports 1100+ languages and NLLB supports 200. To add e.g. Yoruba:

1. Add `"yor": "yor"` to `MMS_LANG` in `server.py`
2. Add `"yor": "yor_Latn"` to `NLLB_SRC`
3. Add an `<option value="yor">Yoruba</option>` to the `<select id="lang">` in
   `static/index.html`
