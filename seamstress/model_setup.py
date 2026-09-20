"""Download the pinned official RIFE model, verifying bytes before publication."""
import io
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile
import hashlib
from .bridge import WEIGHTS_SHA256

MODEL_URL='https://drive.usercontent.google.com/download?id=1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg&export=download&confirm=t'


def setup_model(output):
    output=Path(output)
    if output.exists():
        if hashlib.sha256(output.read_bytes()).hexdigest()!=WEIGHTS_SHA256:raise ValueError('Existing model has wrong checksum; choose a new path.')
        return {'weights':str(output.resolve()),'sha256':WEIGHTS_SHA256,'status':'already verified'}
    request=urllib.request.Request(MODEL_URL,headers={'User-Agent':'Seamstress/0.1'})
    with urllib.request.urlopen(request,timeout=120) as response:
        data=response.read(64*1024*1024+1)
    if len(data)>64*1024*1024:raise ValueError('Model archive exceeds expected size limit.')
    if not zipfile.is_zipfile(io.BytesIO(data)):raise ValueError('Model download was not a ZIP archive; retry the official download.')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names=[n for n in archive.namelist() if n.endswith('/flownet.pkl') or n=='flownet.pkl']
        if len(names)!=1:raise ValueError('Official archive did not contain exactly one checkpoint.')
        if archive.getinfo(names[0]).file_size>64*1024*1024:raise ValueError('Checkpoint exceeds size limit.')
        weights=archive.read(names[0])
    if hashlib.sha256(weights).hexdigest()!=WEIGHTS_SHA256:raise ValueError('Downloaded checkpoint checksum does not match pinned RIFE 4.25.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent,delete=False) as f:
        temp=Path(f.name);f.write(weights)
    try:os.link(temp,output)
    finally:temp.unlink(missing_ok=True)
    return {'weights':str(output.resolve()),'sha256':WEIGHTS_SHA256,'status':'downloaded and verified'}
