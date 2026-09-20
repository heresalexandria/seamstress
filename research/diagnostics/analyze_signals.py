import re,json,statistics,pathlib
p=pathlib.Path(__file__).parent
rows=[]
for line in (p/'signalstats.txt').read_text().splitlines():
    if line.startswith('frame:'):
        m=re.match(r'frame:(\d+)\s+pts:\d+\s+pts_time:(\S+)',line)
        rows.append({'frame':int(m[1]),'time':float(m[2])})
    elif '=' in line:
        k,v=line.split('='); rows[-1][k.rsplit('.',1)[-1]]=float(v)
for i,r in enumerate(rows):
    r['score']=r['YDIF']+r['UDIF']+r['VDIF']
    r['colorjump']=0 if i==0 else sum(abs(r[k]-rows[i-1][k]) for k in ['YAVG','UAVG','VAVG'])
for i,r in enumerate(rows):
    around=[x['score'] for x in rows[max(1,i-24):i]+rows[i+1:i+25]]
    r['ratio']=r['score']/max(.01,statistics.median(around)) if around else 0
(p/'frame_metrics.json').write_text(json.dumps(rows,indent=2))
for key in ['score','ratio','colorjump']:
    print('\nTOP',key)
    for r in sorted(rows,key=lambda r:r[key],reverse=True)[:35]: print(r['frame'],r['time'],round(r[key],3),round(r['score'],3),round(r['colorjump'],3))
