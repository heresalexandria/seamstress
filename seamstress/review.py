"""Write a local, synchronized seam-review player; no server or upload required."""
from pathlib import Path
import html
import json
import os
from fractions import Fraction
import math


def write_review(source, repaired, plan, destination):
    destination=Path(destination)
    src=html.escape(os.path.relpath(source,destination.parent))
    dst=html.escape(os.path.relpath(repaired,destination.parent))
    metadata=plan['source']
    fps=float(Fraction(str(metadata['fps_fraction']))) if metadata.get('fps_fraction') else float(metadata['fps'])
    if not math.isfinite(fps) or fps<=0:
        raise ValueError('Review frame rate must be finite and positive')
    unresolved=set(plan.get('unresolved_seams',[]))
    seams=json.dumps([{'frame':s['frame'],'time':s.get('time',s['frame']/fps),
                       'unresolved':s['frame'] in unresolved} for s in plan.get('seams',[])]).replace('<','\\u003c')
    status=html.escape(str(plan.get('status','Unreviewed candidate; normal-speed playback review required.')))
    if unresolved:
        status+=' Unresolved geometry joins: '+html.escape(', '.join(str(n) for n in sorted(unresolved)))+'.'
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seamstress · seam review</title><style>
:root{font-family:system-ui,sans-serif;color:#e5eaf3;background:#10141d}body{margin:0;padding:30px;max-width:1600px;margin:auto}h1{font-size:28px;margin-bottom:8px}p{color:#aebbd0;line-height:1.5}button,select{border:1px solid #43516a;background:#202c41;color:white;border-radius:8px;padding:10px 14px;font:inherit;cursor:pointer}button.active{background:#346bd7}.videos{display:grid;grid-template-columns:1fr 1fr;gap:16px}video{width:100%;background:black;border-radius:8px}.label{text-transform:uppercase;letter-spacing:.12em;color:#9fb3d2;font-size:12px;margin:16px 0 8px}.controls,.seams{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0}input[type=range]{width:100%;accent-color:#7bafff}.note{border-left:3px solid #738bba;padding-left:15px;font-size:14px}.status{font-variant-numeric:tabular-nums}@media(max-width:800px){.videos{grid-template-columns:1fr}body{padding:16px}}
</style><h1>Seamstress / join review</h1><p>Choose a boundary, loop it, and inspect the motion at normal speed first. Both players follow the original timeline.</p>
<p class="note" id="plan-status">PLAN_STATUS</p><div class="seams" id="seams"></div><div class="videos"><div><div class="label">Original</div><video id="a" src="SOURCE" preload="auto" playsinline></video></div><div><div class="label">Candidate</div><video id="b" src="REPAIRED" preload="auto" playsinline muted></video></div></div>
<div class="controls"><button id="play">Play / pause</button><button id="back">−1 frame</button><button id="next">+1 frame</button><button id="loop" class="active">Loop on</button><select id="speed"><option value="1">Normal speed</option><option value="0.5">Half speed</option><option value="0.25">Quarter speed</option></select><span class="status" id="status"></span></div><input id="scrub" type="range" min="0" max="100" step="0.001"><p class="note">This is a review candidate. Lower measured mismatch does not prove a seam is invisible. Check character outlines, stationary architecture, edge crops, and camera acceleration. The original file remains unchanged.</p>
<script>const seams=SEAMS,fps=FPS;const a=document.querySelector('#a'),b=document.querySelector('#b');let current=0,loop=true,start=0,end=4;const buttons=[];
function seek(t){a.currentTime=t;b.currentTime=t}function pause(){a.pause();b.pause()}function play(){Promise.allSettled([a.play(),b.play()])}function select(i){current=i;const s=seams[i];start=Math.max(0,s.time-2);end=s.time+2;buttons.forEach((x,j)=>x.classList.toggle('active',i===j));seek(start);document.querySelector('#scrub').min=start;document.querySelector('#scrub').max=end;}
seams.forEach((s,i)=>{let button=document.createElement('button');button.textContent=`${Math.floor(s.time/60)}:${(s.time%60).toFixed(2).padStart(5,'0')}${s.unresolved?' · unresolved':''}`;button.onclick=()=>select(i);document.querySelector('#seams').append(button);buttons.push(button)});
document.querySelector('#play').onclick=()=>a.paused?play():pause();document.querySelector('#back').onclick=()=>{pause();seek(Math.max(start,a.currentTime-1/fps))};document.querySelector('#next').onclick=()=>{pause();seek(Math.min(end,a.currentTime+1/fps))};document.querySelector('#loop').onclick=e=>{loop=!loop;e.target.textContent=loop?'Loop on':'Loop off';e.target.classList.toggle('active',loop)};document.querySelector('#speed').onchange=e=>{a.playbackRate=+e.target.value;b.playbackRate=+e.target.value};document.querySelector('#scrub').oninput=e=>{pause();seek(+e.target.value)};
function tick(){if(loop&&a.currentTime>=end){seek(start)}if(!a.paused&&Math.abs(a.currentTime-b.currentTime)>.045)b.currentTime=a.currentTime;document.querySelector('#scrub').value=a.currentTime;document.querySelector('#status').textContent=`${a.currentTime.toFixed(3)} s · frame ${Math.round(a.currentTime*fps)}`;requestAnimationFrame(tick)}a.addEventListener('loadedmetadata',()=>{if(seams.length)select(0);else{start=0;end=a.duration;document.querySelector('#scrub').max=end}},{once:true});tick();</script></html>'''
    page=page.replace('SOURCE',src).replace('REPAIRED',dst).replace('SEAMS',seams).replace('FPS',str(fps)).replace('PLAN_STATUS',status)
    destination.write_text(page)
    return str(destination.resolve())
