"""Run existing SpeakSign ML stages for local uploaded videos."""
import json,subprocess,sys,threading,traceback
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1]
JOBS=ROOT/"storage"/"jobs"
def process(job_id):
 folder=JOBS/job_id
 status=folder/"status.json"
 def update(**fields):
  state=json.loads(status.read_text());state.update(fields);status.write_text(json.dumps(state))
 try:
  video=next(folder.glob("source.*"))
  cap=cv2.VideoCapture(str(video))
  fps=cap.get(cv2.CAP_PROP_FPS);width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release()
  if not (1<=fps<=240 and width>0 and height>0):raise ValueError("Invalid video metadata")
  stages=[
   ("tracking",["ml.person_tracking.multisigner_landmarks","--input",str(video),"--output",str(folder/"hands")]),
   ("landmarks",["ml.person_tracking.fullbody_landmarks","--input",str(video),"--landmarks",str(folder/"hands/multisigner_landmarks.jsonl"),"--output",str(folder/"fullbody"),"--enable-face","--head-crop-fallback"]),
   ("sequences",["ml.person_tracking.sequence_engine","--input",str(folder/"fullbody/signer_sequences.jsonl"),"--output",str(folder/"sequences")]),
   ("recognition",["ml.recognition.gesturesign_inference","--windows",str(folder/"sequences/temporal_windows.jsonl"),"--width",str(width),"--height",str(height),"--fps",str(fps),"--output",str(folder/"predictions.jsonl")]),
   ("timeline",["ml.recognition.temporal_consistency","--input",str(folder/"predictions.jsonl"),"--fps",str(fps),"--output",str(folder/"results")])]
  for i,(name,args) in enumerate(stages):
   update(state="running",stage=name,progress=i*20)
   with (folder/"processing.log").open("a") as log:
    subprocess.run([sys.executable,"-m",*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  update(state="completed",stage="done",progress=100)
 except Exception as exc:
  with (folder/"processing.log").open("a") as log:traceback.print_exc(file=log)
  update(state="failed",error=str(exc))
def start(job_id):
 threading.Thread(target=process,args=(job_id,),daemon=True).start()
