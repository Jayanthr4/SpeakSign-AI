"""V2.7 temporal agreement: candidate labels, never verified ASL translation."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path

def analyze(rows,fps,min_agreement=2,max_gap_frames=16,min_probability=0.29):
    if fps<=0 or min_agreement<2 or max_gap_frames<0 or not 0<=min_probability<=1:
        raise ValueError("Invalid settings")
    tracks=defaultdict(list)
    for row in rows:
        if int(row["start_frame"])>int(row["end_frame"]):raise ValueError("Invalid interval")
        tracks[int(row["track_id"])].append(row)
    timeline=[]
    summary={"fps":fps,"settings":{"min_agreement":min_agreement,"max_gap_frames":max_gap_frames,"min_probability":min_probability},"tracks":{},"warnings":["Overlapping windows are correlated, not independent evidence.","Stable candidates are not verified ASL signs or continuous translation.","Track IDs may switch across occlusions."]}
    for tid,records in sorted(tracks.items()):
        records.sort(key=lambda r:(int(r["start_frame"]),int(r["end_frame"])))
        groups=[]
        for r in records:
            if r.get("status")!="predicted" or not r.get("prediction"):continue
            score=next((float(x["probability"]) for x in r.get("top3",[]) if x.get("sign")==r["prediction"]),0)
            if score<min_probability:continue
            start,end=int(r["start_frame"]),int(r["end_frame"])
            if groups and groups[-1]["sign"]==r["prediction"] and start<=groups[-1]["end"]+max_gap_frames+1:
                g=groups[-1];g["end"]=max(g["end"],end);g["scores"].append(score)
            else:
                groups.append({"sign":r["prediction"],"start":start,"end":end,"scores":[score]})
        stable=0
        for g in groups:
            if len(g["scores"])<min_agreement:continue
            stable+=1
            timeline.append({"track_id":tid,"candidate_sign":g["sign"],"start_frame":g["start"],"end_frame":g["end"],"start_seconds":round(g["start"]/fps,3),"end_seconds":round(g["end"]/fps,3),"supporting_windows":len(g["scores"]),"mean_model_probability":round(sum(g["scores"])/len(g["scores"]),4),"status":"stable_candidate_unverified"})
        summary["tracks"][str(tid)]={"total_windows":len(records),"statuses":dict(Counter(r.get("status","unknown") for r in records)),"stable_candidate_intervals":stable,"rejected_short_groups":len(groups)-stable}
    timeline.sort(key=lambda r:(r["start_frame"],r["track_id"]))
    summary["total_stable_candidate_intervals"]=len(timeline)
    return timeline,summary

def evaluate(timeline,annotations,min_iou=0.5):
    """One-to-one exact-label interval matching; requires exhaustive annotations."""
    if not 0<min_iou<=1:raise ValueError("Invalid IoU")
    candidates=[]
    for pi,p in enumerate(timeline):
        for ai,a in enumerate(annotations):
            if int(p["track_id"])!=int(a["track_id"]) or p["candidate_sign"]!=a["sign"]:continue
            x,y=int(p["start_frame"]),int(p["end_frame"])
            u,v=int(a["start_frame"]),int(a["end_frame"])
            if x>y or u>v:raise ValueError("Invalid interval")
            intersection=max(0,min(y,v)-max(x,u)+1)
            iou=intersection/(max(y,v)-min(x,u)+1)
            if iou>=min_iou:candidates.append((iou,pi,ai))
    used_p,used_a=set(),set()
    for _,pi,ai in sorted(candidates,reverse=True):
        if pi not in used_p and ai not in used_a:used_p.add(pi);used_a.add(ai)
    tp=len(used_p);fp=len(timeline)-tp;fn=len(annotations)-tp
    return {"true_positives":tp,"false_positives":fp,"false_negatives":fn,"precision":round(tp/(tp+fp),4) if tp+fp else None,"recall":round(tp/(tp+fn),4) if tp+fn else None,"min_iou":min_iou,"warning":"Valid only with exhaustive timed labels and stable track identities."}

def read_jsonl(path):
    with path.open() as stream:return [json.loads(line) for line in stream if line.strip()]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--fps",type=float,required=True)
    p.add_argument("--output",type=Path,default=Path("storage/evaluation_v27"))
    p.add_argument("--min-agreement",type=int,default=2)
    p.add_argument("--max-gap-frames",type=int,default=16)
    p.add_argument("--min-probability",type=float,default=0.29)
    p.add_argument("--annotations",type=Path)
    p.add_argument("--min-iou",type=float,default=0.5)
    args=p.parse_args()
    timeline,summary=analyze(read_jsonl(args.input),args.fps,args.min_agreement,args.max_gap_frames,args.min_probability)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/"candidate_timeline.jsonl").open("w") as f:
        for row in timeline:f.write(json.dumps(row)+"\n")
    if args.annotations:summary["labeled_evaluation"]=evaluate(timeline,read_jsonl(args.annotations),args.min_iou)
    (args.output/"evaluation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(f"Stable unverified candidates: {len(timeline)}")
    print(f"Timeline: {args.output/'candidate_timeline.jsonl'}")
    print(f"Summary: {args.output/'evaluation_summary.json'}")

if __name__=="__main__":main()
