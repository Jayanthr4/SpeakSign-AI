"""Experimental per-track wave-motion cue, not ASL lexical recognition."""
from collections import deque

class WaveTracker:
    def __init__(self):
        self.history={}
    def update(self,track_id:int,hand:list[list[float]],now:float)->dict:
        result={"gesture":None,"status":"uncertain"}
        if len(hand)!=21:return result
        # Hand center, normalized to full camera frame.
        cx=sum(hand[i][0] for i in (0,5,9,13,17))/5
        cy=sum(hand[i][1] for i in (0,5,9,13,17))/5
        scale=max(abs(hand[0][1]-hand[9][1]),0.02)
        h=self.history.setdefault(track_id,deque(maxlen=16))
        h.append((now,cx,cy,scale))
        while h and now-h[0][0]>2.4:h.popleft()
        if len(h)<6:return result
        # Horizontal reversals across substantial excursions, not hand jitter.
        directions=[]
        for a,b in zip(list(h),list(h)[1:]):
            delta=b[1]-a[1]
            if abs(delta)>0.12*max(a[3],b[3]):directions.append((1 if delta>0 else -1,b[0]))
        reversals=sum(directions[i][0]!=directions[i-1][0] for i in range(1,len(directions)))
        span=max(x[1] for x in h)-min(x[1] for x in h)
        if reversals>=2 and span>=0.7*max(x[3] for x in h):
            result.update(gesture="WAVE_LIKE_MOTION",status="candidate")
        return result
