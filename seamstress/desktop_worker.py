"""One request per process, newline-delimited JSON events on stdout."""
from __future__ import annotations
import contextlib,json,signal,sys,traceback
from pathlib import Path


def main():
    output=sys.stdout
    def emit(value):
        output.write(json.dumps(value,allow_nan=False)+'\n');output.flush()
    def progress(value):
        total=value.get('total',0)
        fraction=value.get('fraction',value.get('completed',0)/total if total else 0)
        emit({'type':'progress','stage':value.get('stage','analyze'),'progress':fraction,'message':value.get('message','Working…')})
    def stop(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        request=json.loads(sys.stdin.readline());operation=request.get('operation');args=request.get('args',{})
        credentials=request.get('credentials') or {}
        if not isinstance(credentials,dict):raise ValueError('Invalid credential transport')
        provider_key=credentials.get('openaiApiKey')
        if operation=='segmentationStatus':
            from .segmentation_model import model_status
            emit({'type':'complete','result':model_status()});return 0
        with contextlib.redirect_stdout(sys.stderr):
            from .projects import load_project,set_seams,import_seam_correction
            from .pipeline import prepare_project,run_stage,detect_project
            if operation=='create':
                result=prepare_project(args['source'],args['folder'],progress=progress)
                result=detect_project(result,progress=progress)
            elif operation=='get':result=load_project(Path(args['projectPath']))
            elif operation=='setSeams':result=set_seams(Path(args['projectPath']),args['seams'])
            elif operation=='importSeamCorrection':result=import_seam_correction(Path(args['projectPath']),args['frame'],Path(args['reviewedPath']))
            elif operation=='importReconstruction':result=run_stage(args['projectPath'],'reconstruct',
                options={'frame':args['frame'],'action':'import','reconstruction':{'manifestPath':args['manifestPath']}},progress=progress)
            elif operation=='run':result=run_stage(args['projectPath'],args['stage'],options=args.get('options'),progress=progress,provider_key=provider_key)
            else:raise ValueError('Unknown worker operation')
        emit({'type':'complete','project':result});return 0
    except (KeyboardInterrupt,InterruptedError):
        emit({'type':'cancelled','message':'Operation cancelled'});return 130
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit({'type':'error','error':str(exc)});return 1

if __name__=='__main__':raise SystemExit(main())
