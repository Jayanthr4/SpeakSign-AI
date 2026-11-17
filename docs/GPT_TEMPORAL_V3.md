# SpeakSign v3 — Optional GPT temporal vision

This experimental feature keeps the existing YOLO + MediaPipe + ONNX alphabet and everyday gesture engines. It sends **selected short sequences of cropped signer images** to an OpenAI vision-capable model for a separate, tentative interpretation.

## Local setup

From the repository root:

```bash
conda activate SpeakSign_conda_env
python -m pip install "openai>=1.75,<3"
export OPENAI_API_KEY="your-key-from-the-OpenAI-platform"
export SPEAKSIGN_GPT_ENABLED=1
export SPEAKSIGN_GPT_MODEL=gpt-4.1-mini
python -m uvicorn backend.app:app --reload --port 8000
```

Never place the API key in frontend files, commits, browser environment variables, or logs. Use a shell environment variable or a locally managed secrets file excluded from Git. API access is separately billed.

Run the existing frontend with `cd frontend && npm run dev`.

## Behavior

- Server-side only; **disabled by default**.
- Maintains an independent 8-frame ring buffer for each `(session_id, YOLO track_id)`, sampling at most once per 240 ms.
- After six samples and sufficient hand movement, queues a GPT request in a background worker, with a 3.5-second per-track cooldown and at most one in-flight request per track.
- The live frame endpoint does **not wait** for GPT. Predictions arrive in subsequent frame responses.
- Sends downscaled, JPEG-compressed **person crops** rather than the full frame.
- Returns structured `candidate_sign`, `interpretation`, `uncertain`, `explanation`, and observed `latency_ms`.
- Shows status, errors and last candidate per signer. The UI retains existing alphabet and gesture panels.
- Tracking IDs are provisional. A switched ID creates a new buffer; a deleted session discards its state.

## Limitations and privacy

These predictions are experimental. A general-purpose vision model is **not** a validated ASL translator. Frame sampling may miss fast finger motion; an 8-frame window cannot capture long sentences. Model response times vary with network and provider load. The GPT result is *not* fused into local ONNX predictions.

Enabling the feature transmits signer image crops to OpenAI. Only enable with informed consent from people being recorded. For offline-only operation, leave `SPEAKSIGN_GPT_ENABLED` unset. Consider data retention, API costs and access policies before using real participants.

## Validation checklist

1. With GPT disabled, confirm existing alphabet and gestures still run, and the GPT panel reads `disabled`.
2. With GPT enabled, confirm that stationary hands do not continuously trigger requests.
3. Move one hand horizontally and check `analyzing` followed by a candidate or a clear error.
4. Verify that UI frame updates continue while GPT is processing.
5. Compare two signers' predictions and confirm their track IDs are not mixed.
6. Inspect backend output for API errors and benchmark end-to-end response time locally.

No webcam or API integration test has been performed by the code author in your Mac environment.
