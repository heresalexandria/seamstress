"""Explicit classical-CV proposals, not neural or semantic segmentation.

Dense flow only transports labels and reconstruction candidates. It never
warps the original actor image into another pose or morphs whole frames.
"""
from __future__ import annotations

import cv2
import numpy as np


def gray(rgb):
    return cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)


def match_points(source, target, source_mask=None, target_mask=None):
    detector=cv2.SIFT_create(nfeatures=2400)
    sk,sd=detector.detectAndCompute(gray(source),source_mask)
    tk,td=detector.detectAndCompute(gray(target),target_mask)
    if sd is None or td is None or min(len(sd),len(td))<3:
        return np.empty((0,2)),np.empty((0,2))
    matcher=cv2.BFMatcher()
    forward=matcher.knnMatch(sd,td,k=2); backward=matcher.knnMatch(td,sd,k=2)
    reverse={m.queryIdx:m.trainIdx for pair in backward if len(pair)==2 for m,n in [pair] if m.distance<.78*n.distance}
    pairs=[m for pair in forward if len(pair)==2 for m,n in [pair] if m.distance<.78*n.distance and reverse.get(m.trainIdx)==m.queryIdx]
    return np.asarray([sk[m.queryIdx].pt for m in pairs],np.float32).reshape(-1,2),np.asarray([tk[m.trainIdx].pt for m in pairs],np.float32).reshape(-1,2)


def fit_similarity(p,q,threshold=1.5):
    if len(p)<4:
        return np.eye(3),np.zeros(len(p),bool),float('inf')
    fit,keep=cv2.estimateAffinePartial2D(np.asarray(p,np.float32),np.asarray(q,np.float32),
        method=cv2.RANSAC,ransacReprojThreshold=threshold,maxIters=3000,confidence=.995,refineIters=20)
    if fit is None:
        return np.eye(3),np.zeros(len(p),bool),float('inf')
    m=np.vstack((fit,[0,0,1])); keep=keep.ravel().astype(bool)
    residual=np.linalg.norm(np.column_stack((p,np.ones(len(p))))@m[:2].T-q,axis=1)
    return m,keep,float(np.quantile(residual[keep],.90)) if keep.any() else float('inf')


def motion_clusters(incoming,outgoing,max_layers=4):
    p,q=match_points(incoming,outgoing)
    remaining=np.ones(len(p),bool); clusters=[]
    threshold=max(.8,min(incoming.shape[:2])/500)
    for _ in range(max_layers):
        idx=np.where(remaining)[0]
        if len(idx)<8:break
        m,keep,error=fit_similarity(p[idx],q[idx],threshold)
        if keep.sum()<8:break
        selected=idx[keep]
        clusters.append({'matrix':m,'points':p[selected],'targets':q[selected],
                         'inliers':int(keep.sum()),'p90':error})
        remaining[selected]=False
    return clusters,p,q


def local_flow(reference,current):
    """Current→reference flow for pulling a reference label into current space."""
    h,w=current.shape[:2]; factor=min(1.,640/max(w,h));size=(max(8,round(w*factor)),max(8,round(h*factor)))
    a=cv2.resize(gray(current),size); b=cv2.resize(gray(reference),size)
    flow=cv2.calcOpticalFlowFarneback(a,b,None,.5,4,21,4,7,1.5,0)
    flow=cv2.resize(flow,(w,h));flow[:,:,0]*=w/size[0];flow[:,:,1]*=h/size[1]
    y,x=np.mgrid[:h,:w].astype(np.float32)
    return x+flow[:,:,0],y+flow[:,:,1]


def transport_mask(reference,current,mask):
    x,y=local_flow(reference,current)
    result=cv2.remap(mask.astype(np.float32),x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    # Photometric evidence is a diagnostic only; it cannot prove semantic masks.
    warped=cv2.remap(reference,x,y,cv2.INTER_LINEAR)
    error=np.mean(np.abs(warped.astype(float)-current),axis=2)
    ring=cv2.dilate((result>.05).astype(np.uint8),np.ones((5,5),np.uint8))>0
    confidence=float(np.clip(1-np.median(error[ring])/60,0,1)) if ring.any() else 0.
    return result.clip(0,1),confidence


def segment_cluster(rgb,points):
    """Seeded GrabCut proposal. Every such mask needs semantic review."""
    h,w=rgb.shape[:2]
    if len(points)<3:return np.zeros((h,w),np.float32)
    hull=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(hull,cv2.convexHull(np.rint(points).astype(np.int32)),1)
    radius=max(3,round(min(h,w)*.012));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
    broad=cv2.dilate(hull,kernel,iterations=2)
    labels=np.where(broad,cv2.GC_PR_BGD,cv2.GC_BGD).astype(np.uint8)
    labels[hull>0]=cv2.GC_PR_FGD
    for x,y in points:
        cv2.circle(labels,(round(float(x)),round(float(y))),max(1,radius//2),cv2.GC_FGD,-1)
    if not np.any(labels==cv2.GC_BGD):labels[:2]=cv2.GC_BGD
    try:
        cv2.grabCut(rgb,labels,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),3,cv2.GC_INIT_WITH_MASK)
        return np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.float32)
    except cv2.error:
        return hull.astype(np.float32)


def label_planes(shape,clusters):
    """Nearest coherent feature group; a depth-layer proposal, never a claim."""
    h,w=shape
    if len(clusters)==1:return [np.ones((h,w),np.float32)]
    distances=[]
    for cluster in clusters:
        seeds=np.ones((h,w),np.uint8)
        for x,y in cluster['points']:
            seeds[min(h-1,max(0,round(float(y)))),min(w-1,max(0,round(float(x))))]=0
        distances.append(cv2.distanceTransform(seeds,cv2.DIST_L2,3))
    labels=np.argmin(distances,axis=0)
    return [(labels==i).astype(np.float32) for i in range(len(clusters))]


def recover_background(target,mask,donors):
    """Recover only unoccluded donor pixels; mark all inpainted pixels as unknown.

    donors is [(RGB, actor mask, source frame index), ...]. Registration excludes
    both actors. The first reliable donor providing a pixel wins.
    """
    h,w=mask.shape; hole=mask>.001;remaining=hole.copy();plate=target.copy();records=[]
    target_visible=(~cv2.dilate(hole.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)).astype(np.uint8)*255
    for rgb,actor,n in donors:
        if not remaining.any():break
        donor_visible=(~cv2.dilate((actor>.001).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)).astype(np.uint8)*255
        p,q=match_points(rgb,target,donor_visible,target_visible)
        m,keep,error=fit_similarity(p,q,max(1,min(h,w)/500))
        if keep.sum()<8 or error>2.:continue
        warped=cv2.warpPerspective(rgb,m,(w,h),flags=cv2.INTER_LINEAR)
        available=cv2.warpPerspective(donor_visible.astype(np.float32)/255,m,(w,h),flags=cv2.INTER_LINEAR)>.999
        use=remaining&available
        plate[use]=warped[use];remaining[use]=False
        records.append({'source_frame':int(n),'pixels':int(use.sum()),'inliers':int(keep.sum()),'p90_error':error,'matrix':m.tolist()})
    if remaining.any():
        plate=cv2.inpaint(plate,remaining.astype(np.uint8)*255,max(2,min(h,w)/180),cv2.INPAINT_TELEA)
    # Never replace the currently observed scene outside the actual mask.
    plate[~hole]=target[~hole]
    return plate,remaining,records


def cadence_clock(frames,start):
    """Hold identical drawings at one clock tick; no image retiming."""
    clock={start:float(start)}
    for i in range(1,len(frames)):
        same=np.array_equal(frames[i],frames[i-1])
        clock[start+i]=clock[start+i-1] if same else float(start+i)
    return clock


def gate(n,start,end,frame,clock):
    if n<=start or n>=end-1:return 0.
    c=clock[n]
    value=(c-start)/(frame-1-start) if n<frame else (end-1-c)/(end-1-frame)
    t=float(np.clip(value,0,1))
    return t*t*t*(10+t*(-15+6*t))
