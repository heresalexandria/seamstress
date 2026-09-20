"""Source-only whole-segment camera correction, with no frame synthesis."""
import json
from pathlib import Path
import numpy as np
from seamstress.media import read_frames
from seamstress.camera_bridge import _camera_registration,_similarity,_parameters,_transform

p=json.loads(Path('plans/IYTYT-bridges.json').read_text())
center=np.array([639.5,359.5]);cumulative=np.eye(3);rows=[]
for s in p['seams']:
 n=s['frame'];fs=read_frames('IYTYT.mp4',n-5,10)
 left,right=fs[4],fs[5]
 reg=_camera_registration(left,right)
 r=_similarity(np.array(reg['matrix']),center)
 pre=_camera_registration(fs[0],left)
 preforward=_similarity(np.linalg.inv(np.array(pre['matrix'])),center)
 step=_transform(_parameters(preforward,center)/4,center)
 correction=step@r
 cumulative=cumulative@correction
 row={'frame':n,'match_confidence':reg['confidence'],'local_parameters':_parameters(correction,center).tolist(),'cumulative_parameters':_parameters(cumulative,center).tolist(),'local_matrix':correction.tolist(),'cumulative_matrix':cumulative.tolist()}
 rows.append(row);print(n,'local',row['local_parameters'],'cumulative',row['cumulative_parameters'],flush=True)
Path('research/rigid-segments.json').write_text(json.dumps(rows,indent=2))
