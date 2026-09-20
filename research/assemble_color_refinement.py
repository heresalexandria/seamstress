"""Assemble the reviewed color-only recipe without changing baseline geometry."""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from seamstress.conform import validate_conform_plan

BASELINE_SHA='df5f4dd7ea5f34b992fb88e70092d6e6d976dd793003da8bb0a1393d7d1719c8'
CUTS=[361,722,1083,1444,1805,2166,2527,2888,3240]

def assemble(plan_output,calibration_output):
    paths=[Path(plan_output),Path(calibration_output)]
    if paths[0].resolve()==paths[1].resolve() or any(p.exists() for p in paths):
        raise ValueError('Choose two distinct new output paths')
    baseline_path=ROOT/'plans/IYTYT-eight-joins.json';data=baseline_path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=BASELINE_SHA:raise ValueError('Committed baseline plan has changed')
    plan=json.loads(data);calibration=json.loads((ROOT/'plans/IYTYT-calibration.json').read_text())
    curves=[];records=[]
    for cut in CUTS:
        folder=('local-color-strong' if cut==2527 else 'local-color-directional-fine' if cut==3240 else 'local-color-directional')
        path=ROOT/'research'/folder/f'{cut}-model.json';raw=path.read_bytes();model=json.loads(raw)
        if model['frame']!=cut:raise ValueError(f'Model cut mismatch: {path}')
        curves.append(model);records.append({'frame':cut,'model_file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),
            'response':model['left'].get('response','headroom-v1'),
            'reason':('Matches near-black table; dark compression reviewed against native material evidence.' if cut==2527 else
                      'Narrower color/position neighborhoods distinguish jacket from trousers.' if cut==3240 else
                      'Directional color offsets preserve true black/white while matching saturated materials.')})
    provenance={'baseline_reference':'plans/IYTYT-eight-joins.json','baseline_plan_sha256':BASELINE_SHA,'source_sha256':plan['source_sha256'],
                'geometry_changed':False,'timing_changed':False,'models':records}
    for target in [plan,calibration]:
        target['local_color_curves']=curves
        target['local_color_fit_provenance']=provenance
        target['status']='Color-only refinement candidate; playback review required. Original drawings and timing retained; geometry at frame 2888 remains unresolved.'
    validate_conform_plan(plan,plan['source'])
    for path,value in zip(paths,[plan,calibration]):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:f.write(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'plan':str(paths[0].resolve()),'calibration':str(paths[1].resolve()),'model_count':len(curves)},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-output',type=Path,default=ROOT/'plans/IYTYT-color-refined.json')
    parser.add_argument('--calibration-output',type=Path,default=ROOT/'plans/IYTYT-color-refined-calibration.json')
    args=parser.parse_args();assemble(args.plan_output,args.calibration_output)
