from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,json,numpy as np
from seamstress.analysis import pair_metrics
from seamstress.media import probe
cv2.setNumThreads(2)
cap=cv2.VideoCapture('IYTYT.mp4')
seams=[361,722,1083,1444,1805,2166,2527,2888,3240]
records=[]
for target in seams:
 cap.set(cv2.CAP_PROP_POS_FRAMES,target-24);previous=None;window=[]
 for n in range(target-24,target+25):
  ok,frame=cap.read()
  if not ok:break
  frame=cv2.cvtColor(cv2.resize(frame,(480,270)),cv2.COLOR_BGR2RGB)
  if previous is not None:
   m=pair_metrics(previous,frame);m['frame']=n
   m['normalized']=m['registered_mae']/(1+.15*m['motion_p90'])
   m['normalized_p50']=m['registered_mae']/(1+.2*m['motion_p50'])
   window.append(m)
  previous=frame
 records.append({'target':target,'window':window})
 for key in ['registered_mae','normalized','normalized_p50']:
  print(target,key,[(x['frame'],round(x[key],3)) for x in sorted(window,key=lambda x:x[key],reverse=True)[:5]],flush=True)
Path('research/diagnostics/detector_window_metrics.json').write_text(json.dumps(records,indent=2))
