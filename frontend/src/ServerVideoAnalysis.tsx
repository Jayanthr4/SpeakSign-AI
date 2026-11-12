import {useEffect,useRef,useState} from 'react';
type Job={id:string;state:string;stage:string;progress:number;error?:string};
type Candidate={track_id:number;candidate_sign:string;start_seconds:number;end_seconds:number;supporting_windows:number};
export default function ServerVideoAnalysis(){
 const [file,setFile]=useState<File|null>(null);
 const [job,setJob]=useState<Job|null>(null);
 const [savedJobId]=useState(()=>localStorage.getItem('speaksign-last-job'));
 const [items,setItems]=useState<Candidate[]>([]);
 const [error,setError]=useState('');
 const [log,setLog]=useState('');
 const [artifacts,setArtifacts]=useState<{tracking:boolean;fullbody:boolean;browser_preview:boolean;timeline:boolean}|null>(null);
 const player=useRef<HTMLVideoElement>(null);
 useEffect(()=>{
  if(!job||job.state==='completed'||job.state==='failed')return;
  const timer=window.setInterval(async()=>{
   try{
    const response=await fetch('/api/jobs/'+job.id);
    if(!response.ok)throw Error('Status HTTP '+response.status);
    const next=await response.json() as Job;setJob(next);
    const artifactsResponse=await fetch('/api/jobs/'+job.id+'/artifacts');
    if(artifactsResponse.ok)setArtifacts(await artifactsResponse.json() as {tracking:boolean;fullbody:boolean;browser_preview:boolean;timeline:boolean});
    if(next.state==='completed'){
     const result=await fetch('/api/jobs/'+job.id+'/timeline');
     if(result.ok)setItems(await result.json() as Candidate[]);
    }
   }catch(e){setError(String(e));}
  },1500);
  return()=>window.clearInterval(timer);
 },[job?.id,job?.state]);
 async function submit(){
  if(!file)return;
  setError('');setItems([]);setLog('');setArtifacts(null);
  try{
   const form=new FormData();form.append('file',file);
   const response=await fetch('/api/jobs',{method:'POST',body:form});
   if(!response.ok)throw Error(await response.text());
   const created=await response.json() as {id:string;state:string};
   localStorage.setItem('speaksign-last-job',created.id);
   setJob({...created,stage:'queued',progress:0});
  }catch(e){setError(String(e));}
 }
 async function showLog(){
  if(!job)return;
  const response=await fetch('/api/jobs/'+job.id+'/log');
  if(response.ok)setLog((await response.json() as {tail:string}).tail);
 }
 return <section className="panel" style={{padding:24,marginBottom:24}}>
  <h2>Full pipeline video analysis</h2>
  <p style={{margin:'12px 0'}}>Upload to local FastAPI for multi-signer tracking, landmarks and unverified sign candidates. This is not sentence translation.</p>
  <input type="file" accept=".mp4,.mov,.webm,.mkv" onChange={e=>setFile(e.target.files?.[0]||null)}/>
  <button className="primary" disabled={!file||!!job&&!['completed','failed'].includes(job.state)} onClick={()=>void submit()}>Process video</button>
  {job&&<div style={{marginTop:18}}>
   <strong>{job.state} — {job.stage} ({job.progress}%)</strong>
   <progress style={{display:'block',width:'100%',marginTop:8}} value={job.progress} max={100}/>
   <p>Progress is stage-based, not a frame-by-frame percentage. Long stages can appear stationary.</p>
   {artifacts?.tracking&&<p>Tracking preview generated. {artifacts.fullbody?'Full-body landmarks generated.':'Extracting full-body landmarks…'}</p>}
   {job.state==='completed'&&<><p>Unverified candidate intervals: {items.length}</p>
    <video ref={player} controls playsInline preload="metadata" style={{width:'100%',maxWidth:640}} src={'/api/jobs/'+job.id+'/preview'} onError={()=>setError('Browser playback failed. Check the FFmpeg encoding stage and API response.')}/>
    <p>Processed preview includes signer IDs and landmark overlays. Press Play to review.</p>
    {items.map((x,i)=><button key={i} className="outline small" style={{display:'block',margin:'6px 0'}} onClick={()=>{if(player.current){player.current.currentTime=x.start_seconds;void player.current.play().catch(()=>{});}}}>{x.start_seconds.toFixed(2)}–{x.end_seconds.toFixed(2)}s · Track {x.track_id} · {x.candidate_sign} ({x.supporting_windows} windows)</button>)}
    <button className="outline small" onClick={()=>{const url='/api/jobs/'+job.id+'/timeline';void fetch(url).then(r=>r.json()).then(data=>{const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='speaksign-candidates.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});}}>Download candidate results</button>
   </>}
   {job.state==='failed'&&<><p role="alert">{job.error}</p><button className="outline small" onClick={()=>void showLog()}>Show log</button><pre style={{whiteSpace:'pre-wrap',maxHeight:250,overflow:'auto'}}>{log}</pre></>}
  </div>}
  {error&&<p role="alert">{error}</p>}
 </section>;
}
