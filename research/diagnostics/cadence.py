import cv2,numpy as np,json
from pathlib import Path
cv2.setNumThreads(2)
p=Path(__file__).parent
cap=cv2.VideoCapture('IYTYT.mp4');out=[]
for seam in [361,722,1083,1444,1805,2166,2527,2888,3240]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,seam-40);prev=None;rows=[]
 for f in range(seam-40,min(3347,seam+100)):
  ok,img=cap.read()
  if not ok:break
  g=cv2.GaussianBlur(cv2.cvtColor(cv2.resize(img,(640,360)),cv2.COLOR_BGR2GRAY),(5,5),1.1).astype(np.float32)
  if prev is not None:
   d=np.abs(g-prev)
   rows.append({'frame':f,'offset':f-seam,'mean':float(d.mean()),'p95':float(np.percentile(d,95)),'p99':float(np.percentile(d,99)),'over2':float(np.mean(d>2)),'over5':float(np.mean(d>5))})
  prev=g
 out.append({'seam':seam,'rows':rows})
(p/'cadence_metrics.json').write_text(json.dumps(out,indent=2))
for s in out:
 print('\nSeam',s['seam'])
 for r in s['rows']:
  if 0<r['offset']<19:print(r['offset'],round(r['mean'],3),r['p95'],r['p99'],round(r['over2'],4),round(r['over5'],4))
