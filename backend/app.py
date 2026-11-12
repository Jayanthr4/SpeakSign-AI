"""Local-only SpeakSign metadata API. No ASL translation is performed."""
import os
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DB_PATH = Path(os.getenv("SPEAKSIGN_DB_PATH", str(Path(__file__).resolve().parents[1] / "storage" / "sessions.sqlite3")))

def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db

@asynccontextmanager
async def lifespan(app: FastAPI):
    with connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS sessions (
          id TEXT PRIMARY KEY, created_at TEXT, source TEXT,
          label TEXT, frames INTEGER, both_hands INTEGER, notes TEXT)""")
    yield

app = FastAPI(title="SpeakSign AI", version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173"], allow_methods=["GET","POST"], allow_headers=["*"])

class Session(BaseModel):
    source: Literal["camera","video"]
    label: str = Field(min_length=1, max_length=160)
    frames: int = Field(ge=0)
    both_hands: int = Field(ge=0)
    notes: str = Field(default="", max_length=20000)

@app.get("/api/health")
def health():
    return {"status":"ok","landmarks":"browser","asl_translation":"not_available"}

@app.get("/api/sessions")
def sessions():
    with connect() as db:
        return [dict(row) for row in db.execute("SELECT * FROM sessions ORDER BY created_at DESC LIMIT 100")]

@app.post("/api/sessions", status_code=201)
def save_session(session: Session):
    if session.both_hands > session.frames:
        raise HTTPException(422, "both_hands cannot exceed frames")
    result = {"id":str(uuid.uuid4()),"created_at":datetime.now(timezone.utc).isoformat(), **session.model_dump()}
    with connect() as db:
        db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
          (result["id"],result["created_at"],result["source"],result["label"],result["frames"],result["both_hands"],result["notes"]))
    return result


import json
import uuid as _uuid
from fastapi import UploadFile,File
from fastapi.responses import FileResponse
from backend.video_jobs import JOBS,start
@app.post("/api/jobs",status_code=202)
async def upload_job(file:UploadFile=File(...)):
 ext=Path(file.filename or "").suffix.lower()
 if ext not in {".mp4",".mov",".webm",".mkv"}:raise HTTPException(415,"Unsupported video type")
 job_id=str(_uuid.uuid4());folder=JOBS/job_id;folder.mkdir(parents=True)
 path=folder/("source"+ext);size=0
 try:
  with path.open("wb") as output:
   while chunk:=await file.read(1024*1024):
    size+=len(chunk)
    if size>200*1024*1024:raise HTTPException(413,"Max 200 MB")
    output.write(chunk)
  if not size:raise HTTPException(422,"Empty video")
 except Exception:
  path.unlink(missing_ok=True);folder.rmdir();raise
 finally:await file.close()
 (folder/"status.json").write_text(json.dumps({"id":job_id,"state":"queued","stage":"queued","progress":0}))
 start(job_id)
 return {"id":job_id,"state":"queued"}
def job_folder(job_id):
 try:canonical=str(_uuid.UUID(job_id))
 except ValueError:raise HTTPException(404,"Job not found")
 folder=JOBS/canonical
 if not (folder/"status.json").is_file():raise HTTPException(404,"Job not found")
 return folder
@app.get("/api/jobs/{job_id}")
def job_status(job_id:str):
 return json.loads((job_folder(job_id)/"status.json").read_text())
@app.get("/api/jobs/{job_id}/timeline")
def job_timeline(job_id:str):
 path=job_folder(job_id)/"results/candidate_timeline.jsonl"
 if not path.is_file():raise HTTPException(409,"Timeline not ready")
 return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
@app.get("/api/jobs/{job_id}/preview")
def job_preview(job_id:str):
 folder=job_folder(job_id)
 path=folder/"browser_preview.mp4"
 if not path.is_file():
  original=folder/"fullbody/fullbody_preview.mp4"
  if not original.is_file():raise HTTPException(409,"Preview not ready")
  import shutil,subprocess
  ffmpeg=shutil.which("ffmpeg")
  if not ffmpeg:raise HTTPException(503,"Install ffmpeg: brew install ffmpeg")
  result=subprocess.run([ffmpeg,"-y","-i",str(original),"-an","-c:v","libx264","-preset","veryfast","-pix_fmt","yuv420p","-movflags","+faststart",str(path)],capture_output=True)
  if result.returncode:raise HTTPException(500,"Preview conversion failed; inspect server logs")
 return FileResponse(path,media_type="video/mp4")
@app.get("/api/jobs/{job_id}/log")
def job_log(job_id:str):
 path=job_folder(job_id)/"processing.log"
 return {"tail":path.read_text(errors="replace")[-6000:] if path.is_file() else "Not started"}

# V3.1: browser-compatible preview and live multi-person detection
from fastapi import Request
from backend.live_inference import infer_frame,close_session
@app.post("/api/live/{session_id}/frame")
async def live_frame(session_id:str,request:Request):
 try:canonical=str(_uuid.UUID(session_id))
 except ValueError:raise HTTPException(422,"Invalid session ID")
 data=await request.body()
 if len(data)>2_000_000:raise HTTPException(413,"Frame exceeds 2 MB")
 try:return await __import__("starlette.concurrency",fromlist=["run_in_threadpool"]).run_in_threadpool(infer_frame,canonical,data)
 except ValueError as exc:raise HTTPException(422,str(exc))
@app.delete("/api/live/{session_id}")
def stop_live(session_id:str):
 try:canonical=str(_uuid.UUID(session_id))
 except ValueError:raise HTTPException(422,"Invalid session ID")
 close_session(canonical)
 return {"closed":True}
@app.get("/api/jobs/{job_id}/artifacts")
def job_artifacts(job_id:str):
 folder=job_folder(job_id)
 return {"tracking":(folder/"hands/multisigner_preview.mp4").is_file(),
         "fullbody":(folder/"fullbody/fullbody_preview.mp4").is_file(),
         "browser_preview":(folder/"browser_preview.mp4").is_file(),
         "timeline":(folder/"results/candidate_timeline.jsonl").is_file()}
