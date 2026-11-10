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
