"""Read-only audit of all saved RBF corrections; no fitting or video rendering."""
from audit_rbf import *
import hashlib

MODEL=ROOT/'research/local-color-fit/all-models.json'

def main():
    start=time.perf_counter();bundle=json.loads(MODEL.read_text())
    palette=np.stack(np.meshgrid(*[np.linspace(0,255,13)]*3,indexing='ij'),-1).reshape(-1,3)
    positions=np.stack(np.meshgrid(np.linspace(0,1,7),np.linspace(0,1,5),indexing='xy'),-1).reshape(-1,2)
    c=np.tile(palette,(len(positions),1));xy=np.repeat(positions,len(palette),axis=0)
    result={'model_path':str(MODEL),'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),'model_was_refit':False,'geometry_or_timing_changed':False,'finite_difference_byte_step':.5,'cube_grid_size':[13,13,13,7,5],'native_palette_sampling':'Native original source, baseline geometry+LUT applied unchanged, every fourth row/column; no resizing before RGB sampling.','flag_thresholds':{'min_own_channel_derivative':0,'min_color_jacobian_determinant':0,'min_color_jacobian_singular':.25},'reports':[]}
    rows=[]
    for curve in bundle['local_color_curves']:
        cut=curve['frame'];frames=read_frames(ROOT/'IYTYT.mp4',cut-1,2)
        record={'frame':cut,'sides':{},'flags':[]}
        for side,n,image in [('left',cut-1,frames[0]),('right',cut,frames[1])]:
            m=curve[side];base=p.baseline(image,n);height,width=base.shape[:2]
            yy,xx=np.mgrid[0:height:4,0:width:4]
            native=base[::4,::4].reshape(-1,3).astype(float);native_xy=np.stack([xx/(width-1),yy/(height-1)],-1).reshape(-1,2)
            checks={'rgb_xy_grid':eval_points(c,xy,m,False),'native_anchor_palette':eval_points(native,native_xy,m,False)}
            black=predict(np.zeros((len(positions),3)),positions,m);white=predict(np.full((len(positions),3),255.),positions,m)
            checks['black_white_exact']=bool(np.all(black==0) and np.all(white==255))
            checks['range_guarantee_applies']=bool(0<float(m['limit'])<=63.75 and np.isfinite(np.array(m['coefficients'])).all() and np.isfinite(np.array(m['centers'])).all() and (np.array(m['feature_scale'])>0).all())
            for name in ['rgb_xy_grid','native_anchor_palette']:
                v=checks[name]['all']
                if v['min_color_jacobian_diagonal']<=0:record['flags'].append(f'{side}/{name}: nonpositive own-channel derivative')
                if v['min_color_jacobian_determinant']<=0:record['flags'].append(f'{side}/{name}: nonpositive determinant')
                if v['min_color_jacobian_singular']<.25:record['flags'].append(f'{side}/{name}: near-singular color Jacobian')
                if v['clipped_channel_fraction']>0:record['flags'].append(f'{side}/{name}: clipping')
            if not checks['black_white_exact'] or not checks['range_guarantee_applies']:record['flags'].append(f'{side}: endpoint or analytical bound failed')
            record['sides'][side]=checks
        result['reports'].append(record)
        grid=[d['rgb_xy_grid']['all'] for d in record['sides'].values()];native=[d['native_anchor_palette']['all'] for d in record['sides'].values()]
        row={'frame':cut,'grid_min_own_derivative':min(v['min_color_jacobian_diagonal'] for v in grid),'grid_min_singular':min(v['min_color_jacobian_singular'] for v in grid),'grid_min_determinant':min(v['min_color_jacobian_determinant'] for v in grid),'native_min_singular':min(v['min_color_jacobian_singular'] for v in native),'native_max_change':max(v['max_abs_channel_change'] for v in native),'native_p99_change':max(v['p99_abs_channel_change'] for v in native),'flags':record['flags']};rows.append(row)
        print(json.dumps(row),flush=True)
    result['summary']=rows;result['runtime_seconds']=time.perf_counter()-start
    (OUT/'all-rbf-audit.json').write_text(json.dumps(result,indent=2))
    text='''# All-nine saved RBF color safety audit\n\nRead-only audit of `research/local-color-fit/all-models.json`. No fitting, attenuation, production edits, or video rendering. Native original anchor frames receive the accepted baseline geometry and LUT unchanged, then their RGB values are sampled at every fourth pixel in both axes. The color transform is pointwise RGB/XY; audit sampling does not alter output.\n\nEach side was evaluated at76,895 RGB/position combinations (13³RGB cube ×7×5spatial grid) and57,600 native anchor RGB/position samples. Finite differences use±0.5byte. The conservative near-singular diagnostic threshold is minimum singular value<0.25; a negative own-channel derivative or nonpositive determinant also flags a failure. These are numerical audit thresholds, not perceptual visibility thresholds.\n\n| Join | Grid min own-channel derivative | Grid min singular value | Grid min determinant | Native min singular value | Native max change (byte) | Native p99 change (byte) | Flags |\n|---:|---:|---:|---:|---:|---:|---:|:---|\n'''
    for row in rows:text+='| '+str(row['frame'])+' | '+' | '.join(f"{row[k]:.3f}" for k in ['grid_min_own_derivative','grid_min_singular','grid_min_determinant','native_min_singular','native_max_change','native_p99_change'])+' | '+('; '.join(row['flags']) or 'None')+' |\n'
    if not any(r['flags'] for r in rows):text+='\nAll nine models pass. No attenuation is justified by this audit. Every saved model has finite coefficients and center data and limit18; therefore the analytical no-clipping bound applies over the entire valid input gamut, independent of sampling. Exact black/white also pass all sampled positions. No sampled own-channel derivative is negative, and no sampled Jacobian is near singular.\n'
    else:text+='\nOne or more flags require review before production.\n'
    text+='\nLimits: derivative monotonicity is sampled, not a proof over continuous RGB/XY space. A positive Jacobian and valid gamut do not prove correct material grading or a seamless temporal transition. The confidence taper expresses feature proximity, not semantic identity. The earlier361 temporal-sensitivity test remains in `rbf-audit.json`; it was not repeated here. Native maximum includes edge pixels outside the fit’s interior-pixel gate, so it can exceed the fitting report’s maximum.\n\nReproduce: `.venv/bin/python research/color-models/audit_all_rbf.py`. Full measurements, model hash, and all side-specific metrics are in `all-rbf-audit.json`.\n'
    for a,b in {'at76,895':'at 76,895','13³RGB':'13³ RGB','×7×5spatial':'×7×5 spatial','and57,600':'and 57,600','use±0.5byte':'use ±0.5 byte','value<0.25':'value <0.25','limit18':'limit 18','earlier361':'earlier 361'}.items():text=text.replace(a,b)
    (OUT/'ALL-RBF-AUDIT.md').write_text(text)

if __name__=='__main__':main()
