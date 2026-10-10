# Parakeet helper

`inlay-engine` is the server's persistent Parakeet TDT 0.6B v3 process, using the Parakeet implementation in vendored whisper.cpp. It reads audio files supplied by the server; it never opens a microphone or network connection. Builds use Metal on macOS and CPU or CUDA on Linux. See [server setup](../Server/README.md) for packaging and pinned FP16 model weights.

## Protocol

After loading Parakeet and Silero VAD, the helper emits a `ready` JSON object with a `parakeet.cpp/` engine version. Send one UTF-8 JSON object per line on stdin; replies are flushed JSON lines on stdout. Diagnostics go to stderr without transcript text.

```json
{
  "type": "transcribe",
  "id": "request-1",
  "path": "/absolute/path/to/recording.wav",
  "language": "en",
  "vocabularyTerms": ["Inlay", "SwiftUI", "Metal"]
}
```

- `language` defaults to `en`. Accepts `auto` and the model's 25 supported language codes: `bg`, `hr`, `cs`, `da`, `nl`, `en`, `et`, `fi`, `fr`, `de`, `el`, `hu`, `it`, `lv`, `lt`, `mt`, `pl`, `pt`, `ro`, `sk`, `sl`, `es`, `sv`, `ru`, `uk`. Legacy preferences `ja`, `zh`, `ko`, `hi`, and `ar` also remain accepted for proofreading; they do not expand speech recognition support. Recognition always chooses the spoken language automatically. Results return `language: "auto"` because this runtime exposes no language ID. The server omits detected-language metadata and uses the preference only for Qwen proofreading.
- `vocabularyTerms` remains accepted and validated for protocol compatibility. Parakeet has no vocabulary prompting API: `includedTerms` is empty, `omittedTerms` contains every supplied term, and `tokenCount` / `tokenBudget` are zero. Dictionary replacements and Qwen hints remain available after recognition.
- WAV input must be mono 16 kHz PCM16 or float32, 0.2–180 seconds long. The HTTP server uses a 0.25-second minimum.
- Optional `keepFrom` and `keepUntil` (seconds) recognize the whole file as context but return only words that start inside that range. The server uses them for pieces of a take that is still uploading.
- Progress: `{"type":"progress","id":"request-1","value":0.5}`. Updates depend on the upstream callback; large encodes can spend time without intermediate progress.
- Results contain `type: "result"`, `id`, `text`, audio `duration`, processing `elapsed`, `language`, and hint diagnostics.
- Errors contain `type: "error"`, `message`, and a request `id` when available. Requests are processed sequentially.

`{"type":"speech","id":"request-2","path":"/absolute/path/to/window.wav"}` runs only Silero and returns `probabilities` (one speech probability per 32 ms window), `frameSeconds`, `duration`, and `elapsed`. The server uses it to find natural pauses while a take uploads.

Requests are bounded to 1 MiB. Vocabulary allows at most 8,192 terms, 16 KiB per term, and 384 KiB total. Silence returns a successful empty transcript.

## Decoding and lifecycle

A CPU Silero pass rejects nonspeech (threshold 0.5, minimum speech segment 120 ms). If speech is detected, Parakeet receives the complete recording through `parakeet_full`, with greedy TDT decoding and request state reset. The output includes punctuation and capitalization. The full API supports recordings longer than the model's nominal context; memory use grows with recording length. Speech detection remains probabilistic and can miss quiet speech.

The server keeps the model warm. Terminating the helper cancels active work. `{"type":"quit"}`, stdin EOF, or parent death releases it. Native dependencies and Metal source are embedded in the speech executable. whisper.cpp remains linked for Silero; no Whisper recognition model is loaded.

## Verify

Build with `./scripts/build-server.sh`, then use **Inlay Dev** to record speech and inspect progress, transcripts, and the unavailable recognition-vocabulary status in history. Verify cancellation and subsequent recordings interactively when changing helper lifecycle behavior. See [the development guide](../docs/development.md). Automated test harnesses are not allowed.
