import ServerVideoAnalysis from './ServerVideoAnalysis';
import { useEffect, useRef, useState } from 'react';
import { HandLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
import { Activity, ArrowDownToLine, Camera, CheckCircle2, FileText, Hand, History, Layers3, Pause, Play, Radio, ShieldCheck, UploadCloud, Video } from 'lucide-react';

type View = 'overview'|'live'|'video'|'history'|'models';
type Observation = {t:number; hands:number; labels:string[]; landmarks:{x:number;y:number;z:number}[][]};
type Session = {id:string;created_at:string;source:string;label:string;frames:number;both_hands:number;notes:string};
const nav: {key:View;name:string;icon:typeof Hand}[] = [
  {key:'overview',name:'Overview',icon:Layers3},
  {key:'live',name:'Live Studio',icon:Radio},
  {key:'video',name:'Video Analysis',icon:Video},
  {key:'history',name:'Session History',icon:History},
  {key:'models',name:'Model Lab',icon:Activity}
];
function saveFile(name:string,contents:string,type:string) {
  const url=URL.createObjectURL(new Blob([contents],{type}));
  const a=document.createElement('a');a.href=url;a.download=name;a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
export default function App() {
  const [view,setView]=useState<View>('overview');
  const [status,setStatus]=useState('Models not loaded');
  const [playing,setPlaying]=useState(false);
  const [source,setSource]=useState<'camera'|'video'|null>(null);
  const [filename,setFilename]=useState('');
  const [observations,setObservations]=useState<Observation[]>([]);
  const [notes,setNotes]=useState('');
  const [history,setHistory]=useState<Session[]>([]);
  const [message,setMessage]=useState('');
  const [apiOnline,setApiOnline]=useState(false);
  const video=useRef<HTMLVideoElement>(null);
  const canvas=useRef<HTMLCanvasElement>(null);
  const picker=useRef<HTMLInputElement>(null);
  const detector=useRef<HandLandmarker|null>(null);
  const stream=useRef<MediaStream|null>(null);
  const objectUrl=useRef<string|null>(null);
  const frameId=useRef<number>(0);
  const active=useRef(false);
  const lastTime=useRef(-1);
  const lastVideoFrameTime=useRef(-1);
  const lastSample=useRef(0);
  const samples=useRef<Observation[]>([]);
  const startTime=useRef(0);
  const sourceRef=useRef<'camera'|'video'|null>(null);

  async function refresh() {
    try { const r=await fetch('/api/health');setApiOnline(r.ok); } catch {setApiOnline(false);}
    try { const r=await fetch('/api/sessions');if(r.ok)setHistory(await r.json() as Session[]); } catch {/* API optional for camera */}
  }
  useEffect(()=>{void refresh();return ()=>{active.current=false;cancelAnimationFrame(frameId.current);stream.current?.getTracks().forEach(t=>t.stop());if(objectUrl.current)URL.revokeObjectURL(objectUrl.current);detector.current?.close();};},[]);
  async function loadDetector() {
    if(detector.current)return;
    setStatus('Loading MediaPipe model…');
    try {
      const fileset=await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm');
      detector.current=await HandLandmarker.createFromOptions(fileset,{
        baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',delegate:'CPU'},
        runningMode:'VIDEO',numHands:2,minHandDetectionConfidence:0.5,minHandPresenceConfidence:0.5,minTrackingConfidence:0.5
      });
      setStatus('Hand detector ready');
    } catch(e) {setStatus('Model load failed');throw e;}
  }
  function stop() {
    active.current=false;cancelAnimationFrame(frameId.current);
    stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;
    if(video.current){video.current.pause();video.current.srcObject=null;video.current.removeAttribute('src');video.current.load();}
    if(objectUrl.current){URL.revokeObjectURL(objectUrl.current);objectUrl.current=null;}
    sourceRef.current=null;setSource(null);setPlaying(false);
    const c=canvas.current;c?.getContext('2d')?.clearRect(0,0,c.width,c.height);
  }
  function reset() {samples.current=[];setObservations([]);setNotes('');lastSample.current=0;lastVideoFrameTime.current=-1;startTime.current=performance.now();}
  function tick() {
    const v=video.current,c=canvas.current,d=detector.current;
    if(!active.current||!v||!c||!d)return;
    if(v.readyState>=2&&!v.paused){
      const now=performance.now();
      if(now-lastSample.current>160){
        lastSample.current=now;
        // MediaPipe's graph persists across sources. Always supply a strictly
        // increasing timestamp on one shared monotonic clock, even when a
        // recording starts at 0 or is replayed/seeked.
        const stamp=Math.max(now,lastTime.current+1);
        const isVideo=sourceRef.current==='video';
        const mediaTime=v.currentTime;
        // Avoid sending the same decoded video frame repeatedly.
        if(!isVideo || mediaTime!==lastVideoFrameTime.current){
          lastTime.current=stamp;
          if(isVideo)lastVideoFrameTime.current=mediaTime;
          try {
            const result=d.detectForVideo(v,stamp);
            c.width=v.videoWidth;c.height=v.videoHeight;
            const ctx=c.getContext('2d')!;ctx.clearRect(0,0,c.width,c.height);
            const draw=new DrawingUtils(ctx);
            for(const landmarks of result.landmarks){
              draw.drawConnectors(landmarks,HandLandmarker.HAND_CONNECTIONS,{color:'#40e5bd',lineWidth:3});
              draw.drawLandmarks(landmarks,{color:'#fff',fillColor:'#6b5ce7',radius:3});
            }
            const obs:Observation={t:sourceRef.current==='video'?v.currentTime:(now-startTime.current)/1000,
              hands:result.landmarks.length,
              labels:result.handednesses.map(h=>h[0]?.categoryName||'Unknown'),
              landmarks:result.landmarks.map(hand=>hand.map(({x,y,z})=>({x,y,z})))};
            samples.current.push(obs);
            if(samples.current.length>5000)samples.current.shift();
            setObservations([...samples.current]);
          } catch(e){setMessage('Processing error: '+String(e));active.current=false;setPlaying(false);}
        }
      }
    }
    if(active.current)frameId.current=requestAnimationFrame(tick);
  }
  async function cameraStart() {
    stop();reset();setView('live');
    try {
      await loadDetector();
      const s=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});
      stream.current=s;
      const v=video.current!;v.srcObject=s;v.muted=true;await v.play();
      sourceRef.current='camera';setSource('camera');setFilename('Live camera');
      active.current=true;setPlaying(true);frameId.current=requestAnimationFrame(tick);
    }catch(e){stop();setMessage('Camera could not start: '+String(e));}
  }
  async function openVideo(file:File) {
    stop();reset();setView('video');
    try {
      await loadDetector();
      const url=URL.createObjectURL(file);objectUrl.current=url;
      const v=video.current!;v.src=url;v.muted=true;await v.play();
      sourceRef.current='video';setSource('video');setFilename(file.name);
      active.current=true;setPlaying(true);frameId.current=requestAnimationFrame(tick);
    }catch(e){stop();setMessage('Video could not open: '+String(e));}
  }
  function toggle() {
    const v=video.current;if(!v||!source)return;
    if(playing){active.current=false;cancelAnimationFrame(frameId.current);v.pause();setPlaying(false);}
    else {void v.play().then(()=>{active.current=true;setPlaying(true);frameId.current=requestAnimationFrame(tick);}).catch(e=>setMessage(String(e)));}
  }
  async function saveSession() {
    try {
      const payload={source:source||'video',label:filename||'Untitled',frames:observations.length,both_hands:observations.filter(x=>x.hands===2).length,notes};
      const r=await fetch('/api/sessions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      if(!r.ok)throw new Error('HTTP '+r.status);
      setMessage('Session saved to local SQLite database');void refresh();
    }catch(e){setMessage('Could not save: start FastAPI on port 8000. '+String(e));}
  }
  const last=observations.at(-1);
  const both=observations.filter(o=>o.hands===2).length;
  const inStudio=view==='live'||view==='video';
  return <div className="layout">
    <aside className="sidebar"><div className="brand"><span className="brand-icon"><Hand size={22}/></span><div><strong>SpeakSign</strong><small>AI VISION PLATFORM</small></div></div>
      <p className="nav-caption">WORKSPACE</p><nav>{nav.map(({key,name,icon:Icon})=><button key={key} className={view===key?'selected':''} onClick={()=>{if(inStudio&&view!==key)stop();setView(key);if(key==='history')void refresh();}}><Icon size={18}/><span>{name}</span></button>)}</nav>
      <div className="sidebar-bottom"><ShieldCheck size={19}/><div><strong>Local-first by design</strong><p>Video stays on your device.</p></div></div>
      <div className="profile"><span>SS</span><div><strong>Research Workspace</strong><small>ASL · Development</small></div></div>
    </aside>
    <div className="main"><header><div>Workspace <span>/</span> <strong>{nav.find(n=>n.key===view)?.name}</strong></div><div className="api"><i className={apiOnline?'online':''}/>{apiOnline?'Local API online':'API not connected'}</div></header>
      <main>
        {view==='overview'&&<><section className="hero"><div><small className="eyebrow">WELCOME TO YOUR WORKSPACE</small><h1>Making communication<br/><em>more accessible.</em></h1><p>A research-first workspace to capture, understand, and evaluate the visual signals behind signing.</p><div className="actions"><button className="primary" onClick={()=>setView('live')}><Camera size={17}/> Open Live Studio →</button><button className="outline" onClick={()=>setView('video')}><UploadCloud size={17}/> Analyze a video</button></div></div><div className="hero-illustration"><Hand size={94}/></div></section>
          <div className="section-head"><h2>Explore your workspace</h2><p>Tools for accessible, evidence-grounded sign-language research.</p></div>
          <div className="feature-grid">{[{name:'Live Vision Studio',desc:'See up to two hands and their 21 landmarks each, in real time.',icon:Radio,go:'live' as View,tag:'Capture live'},{name:'Video Analysis',desc:'Analyze local recordings and export time-stamped evidence.',icon:Video,go:'video' as View,tag:'Open analyzer'},{name:'Model Research',desc:'Explore the roadmap to continuous ASL translation.',icon:Activity,go:'models' as View,tag:'View roadmap'}].map(f=><button className="feature" key={f.name} onClick={()=>setView(f.go)}><span className="feature-icon"><f.icon size={24}/></span><h3>{f.name}</h3><p>{f.desc}</p><strong>{f.tag} →</strong></button>)}</div>
          <div className="overview-note"><Sparkline/><div><strong>Transparent by design</strong><p>Landmark detection is not ASL translation. We won't generate a sentence until a trained language model has been evaluated.</p></div></div>
        </>}
        {inStudio&&<><div className="page-heading"><small className="eyebrow">{view==='live'?'LIVE VIDEO PIPELINE':'LOCAL VIDEO WORKSPACE'}</small><h1>{view==='live'?'Live Vision Studio':'Video Analysis'}</h1><p>{view==='live'?'Visualize hand movement as it happens.':'Inspect a recording and export reproducible landmark evidence.'}</p></div>
          {view==='video'&&<ServerVideoAnalysis/>}
          <div className="studio-grid"><div><section className="panel player-panel"><div className="panel-header"><strong><Hand size={18}/> Landmark visualizer</strong><span className="pill">{playing?'● PROCESSING':'○ STANDBY'}</span></div>
            <div className="player"><video ref={video} playsInline onEnded={()=>{active.current=false;setPlaying(false);}}/><canvas ref={canvas}/>
              {!source&&<div className="empty-player"><div className="hand-symbol"><Hand size={52}/></div><h3>Your vision workspace</h3><p>{view==='live'?'Connect your webcam to start tracking.':'Choose a video to inspect its hand movements.'}</p><button className="primary" onClick={()=>view==='live'?void cameraStart():picker.current?.click()}>{view==='live'?<Camera size={17}/>:<UploadCloud size={17}/>} {view==='live'?'Enable Camera':'Select Video'}</button></div>}
            </div><div className="player-footer"><div><button className="icon-btn" disabled={!source} onClick={toggle}>{playing?<Pause size={17}/>:<Play size={17}/>}</button><span>{filename||'No source connected'}</span></div><button className="outline small" onClick={()=>view==='live'?void cameraStart():picker.current?.click()}>{view==='live'?'Restart camera':'Choose file'}</button></div></section>
            <section className="panel info-strip"><CheckCircle2 size={19}/><div><strong>{status}</strong><p>Up to two hands · 21 landmarks per hand · no ASL predictions</p></div></section>
          </div><div><section className="panel insight"><div className="panel-header"><strong><Activity size={18}/> Detection insights</strong></div><p>Real frame-level measurements, not sign predictions.</p><div className="numbers"><div><small>Hands visible</small><strong>{last?.hands||0}<em> / 2</em></strong></div><div><small>Frames sampled</small><strong>{observations.length}</strong></div></div>
              <div className="metric"><span>Left hand</span><strong className={last?.labels.includes('Left')?'good':''}>{last?.labels.includes('Left')?'Detected':'Not detected'}</strong></div><div className="metric"><span>Right hand</span><strong className={last?.labels.includes('Right')?'good':''}>{last?.labels.includes('Right')?'Detected':'Not detected'}</strong></div>
              <div className="progress-label"><span>Two-hand visibility</span><strong>{observations.length?Math.round(both/observations.length*100):0}%</strong></div><div className="progress"><div style={{width:(observations.length?both/observations.length*100:0)+'%'}}/></div><small>{both} of {observations.length} frames</small>
            </section><section className="panel notes"><div className="panel-header"><strong><FileText size={18}/> Interpretation workspace</strong></div><div className="warning">ASL sentence recognition is not available yet. These notes are written by you, not generated by AI.</div><label htmlFor="notes">Manual transcript / annotation notes</label><textarea id="notes" placeholder="Write ground-truth observations or transcript notes here…" value={notes} onChange={e=>setNotes(e.target.value)}/>
              <div className="actions"><button className="outline small" disabled={!notes} onClick={()=>saveFile('speaksign-notes.txt',notes,'text/plain')}><ArrowDownToLine size={15}/> Notes .txt</button><button className="outline small" disabled={!observations.length} onClick={()=>saveFile('speaksign-landmarks.json',JSON.stringify({warning:'Not ASL predictions',source:filename,frames:observations},null,2),'application/json')}><ArrowDownToLine size={15}/> Landmarks .json</button></div><button className="primary full" disabled={!observations.length} onClick={()=>void saveSession()}>Save session locally</button></section></div></div>
        </>}
        {view==='history'&&<><div className="page-heading"><small className="eyebrow">LOCAL DATABASE</small><h1>Session History</h1><p>Saved metadata and manual annotations. Videos are not uploaded.</p></div><section className="panel"><div className="panel-header"><strong>Recent sessions</strong><button className="outline small" onClick={()=>void refresh()}>Refresh</button></div>{history.length?history.map(s=><div className="history-item" key={s.id}><div><strong>{s.label}</strong><p>{new Date(s.created_at).toLocaleString()} · {s.frames} frames · {s.both_hands} both-hands frames</p>{s.notes&&<p>{s.notes}</p>}</div><span>{s.source}</span></div>):<p className="muted">No saved sessions yet. Analyze a video and save a session to get started.</p>}</section></>}
        {view==='models'&&<><div className="page-heading"><small className="eyebrow">RESEARCH ROADMAP</small><h1>Model Lab</h1><p>Separating implemented computer vision from future ASL recognition and translation.</p></div>{[{title:'01 · Two-hand landmark extraction',status:'Implemented',text:'MediaPipe Tasks Vision detects up to two hands and exports time-stamped landmark observations.'},{title:'02 · ASL dataset preparation',status:'Planned',text:'Evaluate WLASL and How2Sign usage terms, signer-independent splits and preprocessing.'},{title:'03 · Temporal sign recognition',status:'Planned',text:'Train PyTorch TCN/Transformer models and report held-out WER and accuracy.'},{title:'04 · Grounded English translation',status:'Planned',text:'Translate validated sign sequences and abstain on ambiguous segments.'}].map(x=><section className="panel milestone" key={x.title}><span className={x.status==='Implemented'?'complete':'planned'}>{x.status}</span><h3>{x.title}</h3><p>{x.text}</p></section>)}</>}
      </main><footer>SpeakSign AI · Research Preview <span>Privacy-first · Local development</span></footer>
    </div>
    <input ref={picker} hidden type="file" accept="video/*,.mov,.mp4,.webm" onChange={e=>{const f=e.target.files?.[0];if(f)void openVideo(f);e.target.value='';}}/>
    {message&&<div className="toast">{message}<button onClick={()=>setMessage('')}>×</button></div>}
  </div>;
}
function Sparkline(){return <ShieldCheck size={28}/>;}
