"""Optional Gemini-assisted temporal visual interpretation.

Only server-side code reads GEMINI_API_KEY. Experimental, not validated ASL.
"""
from __future__ import annotations
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import monotonic, perf_counter
import base64
import os
import cv2

_pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix="speaksign-gemini")
_lock=Lock()
_states={}
_MAX_FRAMES=8
_INTERVAL=0.24
_COOLDOWN=3.5
_TTL=120.0

def enabled():
    return bool(os.getenv("GEMINI_API_KEY")) and os.getenv("SPEAKSIGN_GEMINI_ENABLED","0")=="1"

def _interpret(frames):
    """Interpret ordered JPEG crops with Google Gemini. Runs in background pool."""
    from google import genai
    from google.genai import types
    import json
    started=perf_counter()
    prompt=("These are time-ordered webcam images of one person signing, sampled across a short window. "
            "Provide only a cautious possible American Sign Language sign or gesture. "
            "Do not read captions or infer a sentence without clear evidence. "
            "If unclear, use UNKNOWN. Do not invent confidence scores.")
    parts=[types.Part.from_text(text=prompt)]
    parts.extend(types.Part.from_bytes(data=jpg,mime_type="image/jpeg") for jpg in frames)
    client=genai.Client(api_key=os.environ["GEMINI_API_KEY"],http_options=types.HttpOptions(timeout=15000))
    try:
        response=client.models.generate_content(
            model=os.getenv("SPEAKSIGN_GEMINI_MODEL","gemini-2.5-flash-lite"),
            contents=[types.Content(role="user",parts=parts)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type":"OBJECT",
                    "properties":{
                        "candidate_sign":{"type":"STRING"},
                        "interpretation":{"type":"STRING"},
                        "uncertain":{"type":"BOOLEAN"},
                        "explanation":{"type":"STRING"}},
                    "required":["candidate_sign","interpretation","uncertain","explanation"]},
                max_output_tokens=200))
        if not response.text:
            raise RuntimeError("Gemini returned no text (possibly blocked or truncated)")
        parsed=json.loads(response.text)
        parsed["latency_ms"]=round((perf_counter()-started)*1000)
        parsed["model"]=os.getenv("SPEAKSIGN_GEMINI_MODEL","gemini-2.5-flash-lite")
        parsed["status"]="candidate"
        return parsed
    finally:
        client.close()

def update(session_id,track_id,frame,hand,now=None):
    """Nonblocking: submit at most one request per signer on meaningful motion."""
    now=monotonic() if now is None else now
    key=(session_id,int(track_id))
    with _lock:
        # Cleanup only finished/stale states; don't delete active futures.
        for k,v in list(_states.items()):
            if now-v["seen"]>_TTL and (v["future"] is None or v["future"].done()):
                del _states[k]
        state=_states.setdefault(key,{"frames":deque(maxlen=_MAX_FRAMES),"last_sample":0.0,
            "last_trigger":0.0,"last_center":None,"motion":0.0,
            "future":None,"result":None,"seen":now,"error":None})
        state["seen"]=now
        future=state["future"]
        if future is not None and future.done():
            try:state["result"]=future.result();state["error"]=None
            except Exception as exc:
                state["error"]=str(exc)[:200]
            state["future"]=None
        if hand and len(hand)==21:
            center=(sum(p[0] for p in hand)/21,sum(p[1] for p in hand)/21)
            scale=max(abs(hand[0][1]-hand[9][1]),0.025)
            if state["last_center"] is not None:
                delta=((center[0]-state["last_center"][0])**2+(center[1]-state["last_center"][1])**2)**0.5/scale
                state["motion"]=max(delta,state["motion"]*0.8)
            state["last_center"]=center
        if now-state["last_sample"]>=_INTERVAL:
            # Person crop, not entire frame: reduces data and isolates signers.
            resized=cv2.resize(frame,(min(320,frame.shape[1]),max(1,round(frame.shape[0]*min(320,frame.shape[1])/frame.shape[1]))))
            ok,encoded=cv2.imencode(".jpg",resized,[cv2.IMWRITE_JPEG_QUALITY,58])
            if ok:
                state["frames"].append(encoded.tobytes())
                state["last_sample"]=now
        if (enabled() and len(state["frames"])>=6 and state["motion"]>=0.35
                and now-state["last_trigger"]>=_COOLDOWN and state["future"] is None):
            state["future"]=_pool.submit(_interpret,list(state["frames"]))
            state["last_trigger"]=now
            state["motion"]=0.0
        return {"status":("disabled" if not enabled() else
                           "analyzing" if state["future"] is not None else
                           "ready" if state["result"] is not None else "waiting_for_motion"),
                "frames_buffered":len(state["frames"]),
                "motion_score":round(state["motion"],2),
                "prediction":state["result"],
                "error":state["error"] if enabled() else None}

def clear_session(session_id):
    with _lock:
        for key in list(_states):
            if key[0]==session_id:
                # Running futures cannot be cancelled reliably, but are detached.
                del _states[key]
