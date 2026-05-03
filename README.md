# Twi / Igbo → English live speech translator

A small web app that captures live microphone audio in Twi or Igbo,
transcribes it with **Meta MMS** (Massively Multilingual Speech, 1000+
languages), and translates the transcript to English with **NLLB-200**.
Everything runs on your phone — no cloud services, no API keys, no
external accounts.

```
mic → MediaRecorder (5s chunks) → FastAPI /transcribe
                                  → ffmpeg decode (16kHz mono)
                                  → MMS speech recognition
                                  → NLLB-200 translation
                                  → JSON {transcript, translation}
```

Browsers don't natively support Twi or Igbo speech recognition (and
OpenAI Whisper doesn't list them either), so the heavy lifting happens
locally via PyTorch.

## Why this stack

- **MMS** is the only freely available ASR model with Twi (`twi`) and
  Igbo (`ibo`) support.
- **NLLB-200** covers both for translation.
- Both are pure PyTorch + HuggingFace `transformers`, so the whole
  pipeline runs offline on Termux.

## Requirements

- Android phone with **Termux** — install from F-Droid; the Play Store
  build is outdated and won't install the packages below.
- ~3 GB free disk for the models (downloaded on first run).
- Patience on the first request: cold model load takes 30–60 s on a
  phone, then each ~5 s clip transcribes + translates in roughly 5–15 s.

## Termux setup (one-time)

```bash
pkg update && pkg upgrade
pkg install python ffmpeg git rust binutils

# PyTorch and numpy ship as Termux packages — pip-installing them from
# source on Android is painful. Use the pkg versions.
pkg install python-torch python-numpy

git clone <this-repo> translator && cd translator
pip install fastapi uvicorn python-multipart transformers sentencepiece

# Keep model downloads inside the project so they're easy to find/delete.
echo 'export HF_HOME=$HOME/translator/hf_cache' >> ~/.bashrc
source ~/.bashrc
```

> The `transformers` install builds `tokenizers` from source (it's a Rust
> crate with no Android wheel). Expect 5–15 minutes and a lot of RAM.
> If the build crashes with OOM, retry with `CARGO_BUILD_JOBS=1 pip
> install transformers`.

## Run

```bash
cd ~/translator
uvicorn server:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` in the phone's browser, pick **Twi** or
**Igbo**, tap **Start**, and talk. The first request triggers a one-time
model download (~2.5 GB total: MMS ~1 GB, NLLB ~600 MB plus tokenizers).

## Notes & honest limits

- **Low-resource languages.** Twi and Igbo have far less training data
  than English/Spanish/Mandarin. Expect transcription errors — especially
  with background noise, multiple speakers, or heavy code-switching.
- **NLLB-distilled-600M** is small and produces serviceable but
  sometimes literal translations. The full NLLB-3.3B is much better but
  too large for a phone.
- **Near-live, not streaming.** Audio is sent in self-contained 3–8
  second chunks (the page restarts the `MediaRecorder` on each interval
  so each blob has full container headers — required for ffmpeg to
  decode it). True word-by-word streaming would need a different model
  architecture and a WebSocket protocol.
- **Phone CPU is the bottleneck.** No cloud, no GPU. Each chunk takes
  several seconds; consider using larger chunks (8 s) to reduce
  per-chunk overhead.
- **Privacy.** Nothing leaves the phone — no API calls of any kind.

## Models used

- Speech: [`facebook/mms-1b-all`](https://huggingface.co/facebook/mms-1b-all)
- Translation: [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)

## Adding more languages

MMS supports 1100+ languages and NLLB supports 200. To add e.g. Yoruba:

1. Add `"yor": "yor"` to `MMS_LANG` in `server.py`.
2. Add `"yor": "yor_Latn"` to `NLLB_SRC`.
3. Add `<option value="yor">Yoruba</option>` to the `<select id="lang">`
   in `static/index.html`.
