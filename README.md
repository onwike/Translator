# Twi / Igbo → English live speech translator

A small web app that captures live microphone audio in Twi or Igbo and
translates it to English as you speak. Speech recognition always runs
on-device with **Meta MMS**; translation uses a **Grok text model** when an
xAI API key is configured (falling back to **NLLB-200** offline), and an
experimental **Grok Live** mode streams audio straight to xAI's realtime
voice API.

## Two modes

### 1. On-device streaming (default)

```
mic → AudioWorklet (500 ms PCM16 frames) → WS /ws/transcribe
      → rolling buffer (≤30 s)
      → MMS speech recognition (re-run over the whole window)
      → Grok chat translation (XAI_API_KEY set) or NLLB-200 (offline)
      → JSON {partial, transcript, translation, engine}
```

Audio streams continuously in 500 ms frames over a WebSocket. The server
keeps up to 30 seconds of audio in a rolling buffer and re-transcribes the
whole window on every pass, so partial results appear within a couple of
seconds and **revise themselves in place** as more context arrives. The
buffer is also the error guard: a failed pass (model hiccup, Grok timeout)
just retries over the same audio on the next pass — nothing is lost. A
segment finalizes when you pause (~0.7 s of silence), when the 30 s cap is
hit, or when you tap Stop.

Works fully offline with no API key: transcription is always local, and
translation falls back to NLLB-200 automatically (the status line shows
which engine produced each line: `grok` or `nllb`).

### 2. Grok Live (experimental)

```
mic → AudioWorklet (500 ms PCM16 frames) → wss://api.x.ai/v1/realtime
      (ephemeral token minted by POST /grok-live/token — the API key
       never reaches the browser)
```

The browser connects directly to xAI's realtime voice API
(`grok-voice-latest`) with instructions to act as a pure simultaneous
interpreter: it hears Twi/Igbo and emits only the English translation as
text. Sub-second latency, no local ML in the loop.

**Honest caveat:** Igbo and Twi are **not** on Grok's official language
list (~20+ languages for the voice agent, 24 for its STT). This mode
exists to test empirically whether `grok-voice-latest` understands them
anyway. If it doesn't, use on-device streaming — MMS is essentially the
only ASR that genuinely supports both languages. Record your findings
here after testing.

## xAI / Grok setup (optional)

1. Create an API key at [console.x.ai](https://console.x.ai).
2. Put it in `.env` next to `server.py` (already gitignored):

   ```
   XAI_API_KEY=xai-...
   # optional overrides:
   # GROK_TEXT_MODEL=grok-4.3
   # GROK_VOICE_MODEL=grok-voice-latest
   ```

3. Restart the server. `GET /healthz` should now report `"grok": true`
   and the Grok Live option becomes selectable in the UI.

**Costs** (xAI pricing, mid-2026): Grok Live is $0.05/min ($3.00/hr) of
open session. Grok text translation in on-device mode is negligible —
well under $0.01 per hour of speech at $1.25/$2.50 per million tokens.
If partial retranslations burn too many tokens for your taste, the knob
is in `_transcribe_worker` (`server.py`): translate partials with NLLB
and reserve Grok for finalized segments.

## Why this stack

- **MMS** (`facebook/mms-1b-all`) is the only freely available ASR model
  with Twi (`twi`) and Igbo (`ibo`) support. Whisper and Grok's STT don't
  list either language.
- **NLLB-200** covers both for translation, fully offline.
- **Grok** translates noisy ASR transcripts more fluently than
  NLLB-distilled-600M and uses the preceding lines as context.

## Requirements

- Android phone with **Termux** — install from F-Droid; the Play Store
  build is outdated and won't install the packages below.
- ~3 GB free disk for the models (downloaded on first run).
- Patience on the first start: cold model load takes 30–60 s on a phone.

## Termux setup (one-time)

```bash
pkg update && pkg upgrade
pkg install python git rust binutils

# PyTorch and numpy ship as Termux packages — pip-installing them from
# source on Android is painful. Use the pkg versions.
pkg install python-torch python-numpy

git clone <this-repo> translator && cd translator
pip install fastapi uvicorn transformers sentencepiece httpx websockets

# Keep model downloads inside the project so they're easy to find/delete.
echo 'export HF_HOME=$HOME/translator/hf_cache' >> ~/.bashrc
source ~/.bashrc
```

> The `transformers` install builds `tokenizers` from source (it's a Rust
> crate with no Android wheel). Expect 5–15 minutes and a lot of RAM.
> If the build crashes with OOM, retry with `CARGO_BUILD_JOBS=1 pip
> install transformers`.

> ffmpeg is no longer needed — audio now arrives as raw PCM over the
> WebSocket instead of browser-container blobs.

## Run

```bash
cd ~/translator
uvicorn server:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` in the phone's browser, pick **Twi** or
**Igbo**, pick a mode, tap **Start**, and talk. The first start triggers
a one-time model download (~2.5 GB total).

> Note: microphone capture requires a secure context. `http://localhost`
> qualifies; if you serve the page to another device, you'll need HTTPS.

## Notes & honest limits

- **Low-resource languages.** Twi and Igbo have far less training data
  than English/Spanish/Mandarin. Expect transcription errors — especially
  with background noise, multiple speakers, or heavy code-switching.
- **Partials are drafts.** Because the whole rolling window is
  re-transcribed on each pass, the greyed italic line legitimately changes
  as more audio arrives; only finalized lines are stable.
- **Phone CPU is the bottleneck** in on-device mode. The worker is
  adaptive — it processes the latest buffer as soon as the previous pass
  finishes — so on a slow CPU partials simply refresh less often; audio is
  never dropped.
- **Privacy.** On-device mode with no API key: nothing leaves the phone.
  With `XAI_API_KEY` set, transcripts (text) go to xAI for translation;
  in Grok Live mode, raw audio goes to xAI.

## Models used

- Speech: [`facebook/mms-1b-all`](https://huggingface.co/facebook/mms-1b-all)
- Translation: [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)
  or a Grok text model (default `grok-4.3`)
- Grok Live: `grok-voice-latest` via `wss://api.x.ai/v1/realtime`

## Adding more languages

MMS supports 1100+ languages and NLLB supports 200. To add e.g. Yoruba:

1. Add `"yor": "yor"` to `MMS_LANG` in `server.py`.
2. Add `"yor": "yor_Latn"` to `NLLB_SRC`.
3. Add `"yor": "Yoruba"` to `LANG_NAME` and `"yor": "yo"` to `BCP47_HINT`.
4. Add `<option value="yor">Yoruba</option>` to the `<select id="lang">`
   in `static/index.html`.
