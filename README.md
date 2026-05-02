# Twi / Igbo → English live speech translator

A small web app that captures live microphone audio in Twi or Igbo,
transcribes it with **Meta MMS** running on-device, and translates the
transcript to English using **Claude Haiku 4.5** via the Anthropic API.

```
mic → MediaRecorder (15s chunks) → FastAPI /transcribe
                                  → ffmpeg decode (16kHz mono)
                                  → MMS speech recognition (local)
                                  → Claude Haiku 4.5 translation (API)
                                  → JSON {transcript, translation}
```

Runs entirely on an Android phone using **Termux**: the FastAPI server,
PyTorch, and the MMS model all run on the phone; the only network call
goes from the phone to the Anthropic API for translation. Open the page
in the phone's browser at `http://localhost:8000`.

## Why this stack

- Browsers don't natively support Twi or Igbo speech recognition, and
  OpenAI Whisper doesn't list either language. **MMS** covers 1100+
  languages including `twi` and `ibo` and is the only practical free
  option.
- Translation could be done with NLLB-200 locally, but Claude Haiku 4.5
  produces much more natural, context-aware English from a multi-sentence
  Twi/Igbo transcript — and it's fast enough (typically 0.5–2 s per
  chunk).

## Requirements

- Android phone with Termux (install from F-Droid; the Play Store build
  is outdated)
- ~3 GB free disk for the MMS model + PyTorch
- An Anthropic API key (set `ANTHROPIC_API_KEY`)

## Termux setup (one-time)

```bash
pkg update && pkg upgrade
pkg install python ffmpeg git rust binutils

# PyTorch + numpy ship as Termux packages — pip-installing them fails
# without these.
pkg install python-torch python-numpy

git clone <this-repo> translator && cd translator
pip install fastapi 'uvicorn[standard]' python-multipart transformers anthropic

# Keep model downloads inside the project so they're easy to find/delete.
echo 'export HF_HOME=$HOME/translator/hf_cache' >> ~/.bashrc
echo 'export ANTHROPIC_API_KEY=sk-ant-...' >> ~/.bashrc
source ~/.bashrc
```

## Run

```bash
cd ~/translator
uvicorn server:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000` in the phone's browser, pick **Twi** or
**Igbo**, tap **Start**, and talk. The first request triggers a one-time
MMS download (~2.5 GB) and warmup; subsequent chunks transcribe in a few
seconds and translate in ~1 second.

The default 15-second chunk gives Claude enough context to produce
fluent translations across sentence boundaries; drop to 5 s if you'd
rather see English text appear sooner at the cost of choppier phrasing.

## Notes & honest limits

- **Low-resource speech recognition.** Twi and Igbo have far less
  training data than English/Spanish/Mandarin. Expect transcription
  errors — especially with background noise, multiple speakers, or
  heavy code-switching. Claude often makes the English readable even
  when the MMS transcript has minor errors, but garbage in still means
  garbage out.
- **Phone CPU is the bottleneck.** MMS is ~1B params; a 15-second clip
  takes roughly 5–15 s to transcribe on a modern phone. Translation is
  fast.
- **Privacy.** Audio never leaves the phone. The Twi/Igbo *transcript*
  is sent to the Anthropic API for translation — be aware of that if
  the content is sensitive.

## Adding more languages

MMS supports 1100+ languages. To add e.g. Yoruba:

1. Add `"yor": "yor"` to `MMS_LANG` and `"yor": "Yoruba"` to `LANG_NAME`
   in `server.py`.
2. Add `<option value="yor">Yoruba</option>` to the `<select id="lang">`
   in `static/index.html`.

Claude already understands hundreds of languages, so the translation
side needs no changes.
