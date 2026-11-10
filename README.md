# SpeakSign AI — Research Preview

Local-first ASL computer vision research workspace. **This version detects hand landmarks; it does not translate ASL into English.** Translation requires a separately trained and evaluated temporal model.

## Run on macOS

Requires Python 3.11+, Node.js 20+ and internet access for the initial MediaPipe model download.

Terminal 1:
```bash
git clone https://github.com/Jayanthr4/SpeakSign-AI.git
cd SpeakSign-AI
git switch feat/speaksign-v2-foundation
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app:app --reload --port 8000
```

Terminal 2:
```bash
cd SpeakSign-AI/frontend
npm install
npm run dev
```

Open http://localhost:5173. Backend: http://localhost:8000/docs.

## Implemented

- Responsive light-themed React dashboard
- Live webcam and local video input
- MediaPipe Hand Landmarker tracking up to two hands, with 21 points each
- On-screen landmark overlay and frame sampling statistics
- Download JSON landmark observations and manually written notes
- FastAPI + SQLite local session metadata and manual notes
- Research roadmap for ASL training and evaluation

Videos stay in your browser; MediaPipe's WASM and model assets are downloaded from public CDNs at startup. No automatic sign-to-text prediction is provided.

## Roadmap

1. Review licenses and signer-independent splits for WLASL and How2Sign.
2. Build PyTorch isolated sign baseline, evaluate accuracy and confusion matrix.
3. Train and evaluate continuous temporal recognition; report WER.
4. Develop grounded sign-to-English translation and abstention behavior.
5. Add model cards, reproducible evaluation, CI, and optional Docker/cloud deployment.

Do not upload private recordings or unlicensed model weights to this public repository.
