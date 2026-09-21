"""Persistent desktop/CLI projects and frame-exact seam editing."""
from __future__ import annotations
import copy,hashlib,json,math,os,re,subprocess,tempfile,uuid
from pathlib import Path
from fractions import Fraction
from datetime import datetime,timezone
from .media import probe,_tool,_run
from .repair import fingerprint
from .corrections import normalize_correction


def timecode_to_frame(value: str, fps: float | str) -> int:
    """Seconds, MM:SS.mmm, HH:MM:SS.mmm, or non-drop HH:MM:SS:FF."""
    rate=Fraction(str(fps));text=str(value).strip()
    if not text or text.startswith('-'):raise ValueError('Timecode must be nonnegative')
    parts=text.split(':')
    try:
        if len(parts)==4:
            if not all(re.fullmatch(r'\d+',v) for v in parts):raise ValueError()
            h,m,s,f=map(int,parts)
            if m>=60 or s>=60 or f>=round(float(rate)):raise ValueError()
            return (h*3600+m*60+s)*round(float(rate))+f
        if len(parts)>3 or not all(re.fullmatch(r'\d+(?:\.\d+)?',v) for v in parts):raise ValueError()
        if len(parts)>1 and (any('.' in v for v in parts[:-1]) or any(Fraction(v)>=60 for v in parts[1:])):raise ValueError()
        seconds=Fraction(0)
        for p in parts:seconds=seconds*60+Fraction(p)
        return round(seconds*rate)
    except (ValueError,ZeroDivisionError):
        raise ValueError('Use seconds, MM:SS.mmm, HH:MM:SS.mmm or HH:MM:SS:FF (non-drop)') from None


def atomic_json(path: Path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.seamstress-',suffix='.json',delete=False) as f:
        temporary=Path(f.name)
        try:
            json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True);raise
    os.replace(temporary,path)


def normalized_seams(seams,metadata,source_sha256=None):
    if not isinstance(seams,list) or len(seams)>10000:raise ValueError('Seams must be a list of at most 10,000 boundaries')
    seen=set();result=[]
    for value in seams:
        row={'frame':value} if type(value) is int else dict(value)
        frame=row.get('frame')
        if type(frame) is not int or not 0<frame<metadata['frame_count']:raise ValueError('A seam must be the first incoming frame, inside the video')
        if frame in seen:continue
        seen.add(frame)
        confidence=row.get('confidence')
        if confidence is not None and (type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1):raise ValueError('Seam confidence must be between 0 and 1')
        if 'enabled' in row and type(row['enabled']) is not bool:raise ValueError('Seam enabled must be true or false')
        correction=normalize_correction(row.get('correction'),metadata=metadata,source_sha256=source_sha256,frame=frame)
        result.append({**row,'id':str(row.get('id') or f'seam-{frame}-{uuid.uuid4().hex[:6]}'),
            'frame':frame,'time':frame/metadata['fps'],'enabled':row.get('enabled',True),'origin':row.get('origin','manual'),
            'kind':row.get('kind',row.get('classification','continuation')),'correction':correction})
    return sorted(result,key=lambda r:r['frame'])


def inspect_source(source:Path,progress=None,cancelled=None):
    """Validate display geometry and actual frame timestamps before correction."""
    meta=probe(source)
    if abs(meta.get('video_start_time',0))>.0002:
        raise ValueError('Video begins at a nonzero media timestamp. Re-export with the video timeline starting at zero before importing, to preserve audio synchronization')
    if meta['frame_count']<2:raise ValueError('Choose a video containing at least two frames')
    if min(meta['width'],meta['height'])<32 or meta['width']%2 or meta['height']%2:
        raise ValueError('This version supports even video dimensions of at least 32 pixels; re-export this source at an even resolution')
    pixel_formats=json.loads(_run([_tool('ffprobe'),'-v','error','-show_pixel_formats','-of','json']).stdout)['pixel_formats']
    pixel_format=next((p for p in pixel_formats if p['name']==meta.get('pix_fmt')),None)
    depth=max((c['bit_depth'] for c in (pixel_format or {}).get('components',[])),default=0)
    if meta.get('color_transfer') in ('smpte2084','arib-std-b67') or depth>8:
        raise ValueError('This version processes 8-bit SDR video. Convert HDR/high-bit-depth footage to SDR before importing')
    if not depth:raise ValueError('Cannot establish source pixel depth; convert this source to 8-bit SDR first')
    sar=meta.get('sample_aspect_ratio')
    try:square_pixels=sar in (None,'N/A','0:1') or Fraction(sar.replace(':','/'))==1
    except (ValueError,ZeroDivisionError):square_pixels=False
    if not square_pixels:
        raise ValueError('Anamorphic (non-square-pixel) video is not supported yet. Re-export with square pixels before importing')
    if meta.get('color_space') not in (None,'unknown','unspecified','bt709','bt470bg','smpte170m','smpte240m','fcc','gbr'):
        raise ValueError('This SDR color matrix is not supported yet; convert to BT.709 SDR before importing')
    if progress:progress({'stage':'import','fraction':.05,'message':'Checking frame timing and orientation'})
    args=[_tool('ffprobe'),'-v','error','-select_streams',str(meta.get('video_stream_index',0)),
          '-show_entries','frame=best_effort_timestamp_time','-of','csv=p=0',str(source)]
    process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    previous=None;first_stamp=None;count=0;irregular=0;period=1/meta['fps']
    try:
        for line in process.stdout:
            if cancelled and cancelled():raise InterruptedError('Import cancelled')
            try:stamp=float(line.strip().split(',')[0])
            except ValueError:continue
            if first_stamp is None:first_stamp=stamp
            if previous is not None and abs((stamp-previous)-period)>max(.0002,period*.025):irregular+=1
            previous=stamp;count+=1
            if progress and count%max(120,meta['frame_count']//100)==0:
                progress({'stage':'import','fraction':.05+.6*count/meta['frame_count'],
                          'message':f'Checking frame timing: {count:,} / {meta["frame_count"]:,}'})
        error=process.stderr.read();code=process.wait()
        if code:raise RuntimeError(error.strip() or 'Could not inspect frame timestamps')
    finally:
        if process.poll() is None:process.terminate();process.wait()
        process.stdout.close();process.stderr.close()
    if count!=meta['frame_count']:raise ValueError('Video frame count disagrees with decoded timestamps; re-export a constant-frame-rate source')
    if first_stamp is None or abs(first_stamp)>.0002:raise ValueError('Video frame timestamps must begin at zero; re-export with a zero-based timeline to preserve audio synchronization')
    if irregular:raise ValueError('Variable or discontinuous frame timing detected. Re-export at a constant frame rate before importing; Seamstress will not silently change timing')
    return meta


def create_project(source:Path,project_dir:Path,*,progress=None,cancelled=None):
    source=Path(source).expanduser().resolve();folder=Path(project_dir).expanduser().resolve();path=folder/'project.json'
    if path.exists():raise FileExistsError('A project already exists here; reopen it or choose a new directory')
    meta=inspect_source(source,progress,cancelled)
    if progress:progress({'stage':'import','fraction':.7,'message':'Identifying the source video'})
    digest=fingerprint(source)
    project={'version':1,'id':uuid.uuid4().hex,'name':source.stem,'projectPath':str(path),'source':str(source),
             'sourceSha256':digest,'sourceStat':{'size':source.stat().st_size,'mtimeNs':source.stat().st_mtime_ns},
             'metadata':meta,'seams':[],'revision':0,'artifacts':{},'status':'imported','warnings':[],
             'createdAt':datetime.now(timezone.utc).isoformat(),'updatedAt':datetime.now(timezone.utc).isoformat()}
    atomic_json(path,project)
    return project


def load_project(path:Path,*,check_source=True):
    path=Path(path).expanduser().resolve()
    if path.is_dir():path=path/'project.json'
    data=json.loads(path.read_text())
    if not isinstance(data,dict) or data.get('version')!=1 or not isinstance(data.get('artifacts'),dict):raise ValueError('Not a supported Seamstress project')
    if not isinstance(data.get('source'),str) or not isinstance(data.get('metadata'),dict):raise ValueError('Project is missing its source')
    data['projectPath']=str(path);data['seams']=normalized_seams(data.get('seams',[]),data['metadata'],data.get('sourceSha256'))
    if check_source:
        source=Path(data['source'])
        if not source.is_file():raise FileNotFoundError(f'The original video has moved: {source}')
        fresh=probe(source)
        for key in ('sample_aspect_ratio','video_start_time','audio_start_times'):
            data['metadata'][key]=fresh[key]
        if abs(fresh.get('video_start_time',0))>.0002:
            raise ValueError('Video begins at a nonzero media timestamp; re-export on a zero-based timeline to preserve audio synchronization')
        stat=source.stat();cached=data.get('sourceStat',{})
        if stat.st_size!=cached.get('size') or stat.st_mtime_ns!=cached.get('mtimeNs'):
            if fingerprint(source)!=data.get('sourceSha256'):raise ValueError('The source video has changed; create a new project')
    return data


def save_project(project):
    project['updatedAt']=datetime.now(timezone.utc).isoformat()
    atomic_json(Path(project['projectPath']),project)
    return project


def set_seams(path:Path,seams):
    project=load_project(path)
    # An exact-frame measurement is never carried to a newly placed boundary.
    # Apply this in the worker too; callers other than the UI can move markers.
    old_by_id={r['id']:r for r in project['seams']}
    if isinstance(seams,list):
        seams=copy.deepcopy(seams)
        for row in seams:
            if not isinstance(row,dict):continue
            old=old_by_id.get(row.get('id'))
            if old and row.get('frame')!=old['frame'] and isinstance(row.get('correction'),dict):
                correction=row['correction']
                if correction.pop('manual',None) is not None and correction.get('geometry')=='manual':
                    correction['geometry']='auto'
    rows=normalized_seams(seams,project['metadata'],project.get('sourceSha256'))
    def signature(items):
        return [(r['frame'],r['enabled'],r['correction']) for r in items]
    before=signature(project['seams']);after=signature(rows)
    project['seams']=rows
    if before!=after:
        project['revision']+=1
        keep=('proxy','thumbnails','detection')
        project['artifacts']={k:v for k,v in project['artifacts'].items() if k in keep}
        project['status']='marked';project['warnings']=[];project.pop('seamResults',None)
    if project['status']=='imported':project['status']='marked'
    return save_project(project)


def import_seam_correction(path:Path,frame:int,reviewed_path:Path):
    """Copy one source-bound reviewed cut; color is refitted on next analysis.

    Only measurements are imported. No paths or color models from the external
    calibration become executable project inputs.
    """
    from .design import ALGORITHM
    project=load_project(path)
    if type(frame) is not int or not any(row['frame']==frame for row in project['seams']):
        raise ValueError('Select an existing seam before importing reviewed framing')
    reviewed_path=Path(reviewed_path).expanduser().resolve()
    if not reviewed_path.is_file() or reviewed_path.stat().st_size>16*1024*1024:
        raise ValueError('Choose a reviewed calibration JSON smaller than 16 MiB')
    # Limit the read itself too, in case the file grows after stat().
    with reviewed_path.open('rb') as handle:data=handle.read(16*1024*1024+1)
    if len(data)>16*1024*1024:raise ValueError('Reviewed calibration exceeds 16 MiB')
    try:calibration=json.loads(data)
    except (ValueError,UnicodeDecodeError):raise ValueError('Choose a valid reviewed calibration JSON') from None
    if (not isinstance(calibration,dict) or calibration.get('schema_version')!=1 or
            calibration.get('method')!='source_conform_calibration' or calibration.get('algorithm')!=ALGORITHM):
        raise ValueError('Choose a source_conform_calibration JSON, not a render plan or project')
    if calibration.get('source_sha256')!=project['sourceSha256']:
        raise ValueError('Reviewed framing belongs to a different source video (SHA-256 mismatch)')
    metadata=calibration.get('source')
    if not isinstance(metadata,dict) or any(metadata.get(k)!=project['metadata'][k] for k in ('width','height','frame_count','fps_fraction')):
        raise ValueError('Reviewed framing source dimensions or timing differ from this project')
    cuts=calibration.get('cuts');excluded=calibration.get('excluded_geometry',[])
    if (not isinstance(cuts,list) or len(cuts)>10000 or not all(isinstance(cut,dict) for cut in cuts) or
            not isinstance(excluded,list) or not all(isinstance(row,dict) for row in excluded)):
        raise ValueError('Reviewed calibration has invalid cut records')
    matching=[cut for cut in cuts if type(cut.get('frame')) is int and cut['frame']==frame]
    if len(matching)!=1:raise ValueError(f'Reviewed calibration must contain exactly one cut at frame {frame}')
    if any(row.get('frame')==frame for row in excluded):
        raise ValueError(f'Frame {frame} was excluded from this calibration; choose accepted measurements')
    cut=matching[0]
    manual={key:copy.deepcopy(cut.get(key)) for key in ('right_to_left_matrix','pre_rate','post_rate','ease_rate')}
    manual['provenance']={'kind':'imported','label':reviewed_path.name,'frame':frame,
                          'source_sha256':project['sourceSha256'],'calibration_sha256':hashlib.sha256(data).hexdigest()}
    rows=copy.deepcopy(project['seams']);row=next(row for row in rows if row['frame']==frame)
    row['correction']=normalize_correction({**row['correction'],'geometry':'manual','manual':manual,
                                           'rate_easing':manual['ease_rate']},
        metadata=project['metadata'],source_sha256=project['sourceSha256'],frame=frame)
    return set_seams(path,rows)
