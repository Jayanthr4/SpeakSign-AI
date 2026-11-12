import {useEffect,useRef,useState} from 'react';
type AlphabetCue={letter:string|null;status:string;method:string};
type Person={alphabet?:AlphabetCue[];track_id:number;box:number[];hands:number[][][];confidence:number;pose:number[][];face:number[][]};
const POSE_LINKS=[[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],[0,1],[0,4]];
const FACE_LINKS=[[10,338],[338,297],[297,332],[332,284],[284,251],[251,389],[389,356],[356,454],[454,323],[323,361],[361,288],[288,397],[397,365],[365,379],[379,378],[378,400],[400,377],[377,152],[152,148],[148,176],[176,149],[149,150],[150,136],[136,172],[172,58],[58,132],[132,93],[93,234],[234,127],[127,162],[162,21],[21,54],[54,103],[103,67],[67,109],[109,10],[33,133],[362,263],[61,40],[40,37],[37,0],[0,267],[267,270],[270,291]];
const LINKS=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
export default function LiveMultiSigner(){
 const video=useRef<HTMLVideoElement>(null),canvas=useRef<HTMLCanvasElement>(null),capture=useRef<HTMLCanvasElement>(null);
 const session=useRef<string|null>(null),stream=useRef<MediaStream|null>(null),busy=useRef(false);
 const [showFace,setShowFace]=useState(true),[active,setActive]=useState(false),[people,setPeople]=useState<Person[]>([]),[latency,setLatency]=useState<number|null>(null),[message,setMessage]=useState('Start the camera to track people and hands.');
 useEffect(()=>()=>{stop();},[]);
 function stop(){
  const id=session.current;session.current=null;
  if(id)void fetch('/api/live/'+id,{method:'DELETE'}).catch(()=>{});
  stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;
  if(video.current){video.current.srcObject=null;}
  setActive(false);setPeople([]);
 }
 async function start(){
  stop();
  try{
   const media=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480}},audio:false});
   stream.current=media;session.current=crypto.randomUUID();
   if(!video.current){media.getTracks().forEach(t=>t.stop());return;}
   video.current.srcObject=media;await video.current.play();
   setActive(true);setMessage('Running YOLO + hand, pose, and face landmarks. Track IDs may change.');
  }catch(e){setMessage('Camera error: '+String(e));}
 }
 useEffect(()=>{
  if(!active)return;
  const timer=window.setInterval(async()=>{
   const v=video.current,c=capture.current,id=session.current;
   if(!v||!c||!id||busy.current||v.readyState<2)return;
   busy.current=true;
   try{
    c.width=640;c.height=Math.round(v.videoHeight*640/v.videoWidth);
    c.getContext('2d')?.drawImage(v,0,0,c.width,c.height);
    const blob=await new Promise<Blob|null>(resolve=>c.toBlob(resolve,'image/jpeg',0.7));
    if(!blob)return;
    const response=await fetch('/api/live/'+id+'/frame',{method:'POST',headers:{'Content-Type':'image/jpeg'},body:blob});
    if(!response.ok)throw Error('Live inference HTTP '+response.status+': '+await response.text());
    const result=await response.json() as {people:Person[];inference_ms?:number};
    if(session.current===id){setPeople(result.people);setLatency(result.inference_ms??null);}
   }catch(e){setMessage(String(e));}
   finally{busy.current=false;}
  },300);
  return()=>window.clearInterval(timer);
 },[active]);
 useEffect(()=>{
  const c=canvas.current,v=video.current;
  if(!c||!v||!v.videoWidth)return;
  c.width=v.videoWidth;c.height=v.videoHeight;
  const ctx=c.getContext('2d');if(!ctx)return;
  const drawingContext=ctx;
  const canvasWidth=c.width;
  const canvasHeight=c.height;
  ctx.clearRect(0,0,c.width,c.height);
  for(const person of people){
   const [x1,y1,x2,y2]=person.box;
   ctx.strokeStyle='#41e5b9';ctx.lineWidth=3;ctx.strokeRect(x1*c.width,y1*c.height,(x2-x1)*c.width,(y2-y1)*c.height);
   ctx.fillStyle='#41e5b9';ctx.font='bold 18px sans-serif';ctx.fillText('Track '+person.track_id,x1*c.width,Math.max(20,y1*c.height-8));
   function drawEdges(points:number[][],edges:number[][],color:string){
    drawingContext.strokeStyle=color;drawingContext.lineWidth=2;
    for(const [a,b] of edges){
     if(!points[a]||!points[b])continue;
     drawingContext.beginPath();drawingContext.moveTo(points[a][0]*canvasWidth,points[a][1]*canvasHeight);
     drawingContext.lineTo(points[b][0]*canvasWidth,points[b][1]*canvasHeight);drawingContext.stroke();
    }
   }
   drawEdges(person.pose||[],POSE_LINKS,'#42b5ff');
   if(showFace && person.face?.length){
    drawEdges(person.face,FACE_LINKS,'#ff3fa9');
    drawingContext.fillStyle='#ff3fa9';
    for(const point of person.face){
     if(!Number.isFinite(point[0])||!Number.isFinite(point[1]))continue;
     drawingContext.beginPath();
     drawingContext.arc(point[0]*canvasWidth,point[1]*canvasHeight,1.5,0,Math.PI*2);
     drawingContext.fill();
    }
   }
   for(const hand of person.hands){
    ctx.strokeStyle='#ffcc6a';ctx.lineWidth=2;
    for(const [a,b] of LINKS){ctx.beginPath();ctx.moveTo(hand[a][0]*c.width,hand[a][1]*c.height);ctx.lineTo(hand[b][0]*c.width,hand[b][1]*c.height);ctx.stroke();}
    for(const [x,y] of hand){ctx.beginPath();ctx.arc(x*c.width,y*c.height,3,0,Math.PI*2);ctx.fillStyle='#ffffff';ctx.fill();}
   }
  }
 },[people,showFace]);
 return <section className="panel" style={{padding:24,marginBottom:24}}>
  <h2>Live multi-signer tracking</h2>
  <p style={{margin:'12px 0'}}>Experimental YOLO tracking with hand, pose, and face overlays per tracked person. No live ASL recognition or transcription.</p>
  <div className="player" style={{position:'relative',maxWidth:800}}>
   <video ref={video} muted playsInline style={{display:'block',width:'100%',height:'auto'}}/>
   <canvas ref={canvas} style={{position:'absolute',top:0,left:0,width:'100%',height:'100%',display:'block',zIndex:10,pointerEvents:'none'}}/>
  </div>
  <canvas ref={capture} style={{display:'none'}}/>
  <label style={{display:'flex',alignItems:'center',gap:8,marginTop:10}}>
   <input type="checkbox" checked={showFace} onChange={e=>setShowFace(e.target.checked)}/>
   Show face mesh (pink, 478 landmarks)
  </label>
  <div className="actions" style={{marginTop:12}}>
   <button className="primary" onClick={()=>void start()}>{active?'Restart tracking':'Start live tracking'}</button>
   <button className="outline" disabled={!active} onClick={stop}>Stop</button>
  </div>
  <p>{message}</p>
  <p>Inference time: {latency===null?'Waiting for frame…':latency+' ms'} · Isolated-sign recognition: not connected to live camera</p>
  <div style={{padding:12,border:'1px solid #7775',borderRadius:8,marginTop:10}}>
   <strong>Experimental fingerspelling handshape candidates</strong>
   <p>Prototype supports only B, D, I, L, V, Y. Not a trained A–Z recognizer; no J/Z motion recognition. Treat results as unverified.</p>
   <p>{people.flatMap(p=>(p.alphabet||[]).map((cue,i)=>'Track '+p.track_id+' hand '+(i+1)+': '+(cue.letter||'Uncertain'))).join(' · ')||'No hands detected / no letter candidates'}</p>
  </div>
  <p>Face diagnostic: {people.length?people.map(p=>'Track '+p.track_id+': '+(p.face?.length||0)+' face landmarks').join(' · '):'No tracked people yet'}</p>
  <p>People tracked: {people.length} · {people.map(p=>'Track '+p.track_id+' (H'+p.hands.length+' P'+Number((p.pose||[]).length>0)+' F'+Number((p.face||[]).length>0)+')').join(', ')||'No detections'}</p>
 </section>;
}
