# V3 application integration

From repository root, install backend and ML dependencies:

```bash
conda activate SpeakSign_conda_env
python -m pip install -r backend/requirements.txt
python -m pip install -r ml/requirements-tracking.txt -r ml/requirements-landmarks.txt numpy onnxruntime
export PYTHONPATH="$PWD:/tmp/GestureSign/pipeline/src:${PYTHONPATH:-}"
python -m uvicorn backend.app:app --reload --port 8000
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and choose Video Analysis, then **Full pipeline video analysis**. GestureSign model files must already be present in storage/models/gesturesign and upstream GestureSign feature code must be importable.

This is a local prototype. Processing runs in a daemon thread, so restarting the API interrupts jobs. There is no authentication, cancellation, or automatic cleanup. Uploaded videos are stored on the local API server, unlike the browser-only viewer. Model labels are unverified candidates, not ASL translation.
