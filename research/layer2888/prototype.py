"""Experimental source-only foreground/background conform at frame2888."""
from pathlib import Path
import sys,json,cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from seamstress.media import read_frames,VideoWriter
from seamstress.registration import _fit_color
from research.rigid_feasibility import features
from PIL import Image,ImageDraw
cv2.setNumThreads(4)
OUT=ROOT/'research/layer2888'
W,H=1280,720
# Rough native outline annotations of incoming frame2888. They constrain GrabCut;
# no generated imagery/model is used.
SLOTH=[(478,271),(497,269),(518,273),(531,283),(535,310),(530,325),(559,335),(585,330),(613,318),(626,318),(635,326),(627,339),(600,354),(569,367),(555,364),(560,386),(568,402),(571,464),(562,510),(558,529),(546,542),(540,542),(535,535),(529,527),(520,519),(509,478),(498,478),(485,520),(480,536),(469,544),(457,540),(449,527),(442,502),(436,479),(428,450),(431,396),(437,376),(444,367),(422,369),(401,364),(374,353),(348,347),(340,339),(343,330),(350,328),(383,331),(415,337),(438,331),(462,322),(465,296)]
WOMAN=[(702,229),(715,225),(738,228),(749,237),(753,258),(756,274),(752,286),(765,294),(778,309),(794,321),(815,328),(837,332),(849,336),(860,345),(864,352),(858,358),(849,358),(838,352),(813,346),(797,339),(787,340),(786,367),(789,391),(789,423),(801,460),(813,500),(820,536),(831,550),(840,563),(841,578),(834,585),(823,587),(810,579),(795,573),(784,550),(774,541),(769,517),(754,476),(751,523),(751,543),(746,553),(751,564),(753,580),(746,590),(739,598),(730,599),(719,589),(712,575),(709,558),(707,542),(711,461),(710,426),(706,396),(696,391),(692,407),(685,459),(677,480),(669,475),(660,450),(656,407),(657,386),(665,367),(669,347),(655,343),(640,338),(628,334),(620,339),(611,337),(608,330),(612,322),(621,316),(632,321),(645,321),(658,315),(672,302),(682,294),(689,273),(693,247)]
BASE=np.zeros((H,W),np.uint8)
for p in [SLOTH,WOMAN]:cv2.fillPoly(BASE,[np.array(p,np.int32)],1)
# Open air between sloth legs and other gaps are left to color segmentation;
# explicitly remove the known large gap between its legs.
cv2.fillPoly(BASE,[np.array([(486,521),(498,478),(509,478),(520,519),(514,537),(490,537)],np.int32)],0)

def grab(frame,seed=BASE):
 mask=np.zeros((H,W),np.uint8)
 dil=cv2.dilate(seed,np.ones((19,19),np.uint8))>0
 inner=cv2.erode(seed,np.ones((11,11),np.uint8))>0
 mask[dil]=cv2.GC_PR_BGD;mask[seed>0]=cv2.GC_PR_FGD
 # Definite foreground only on colored subject interiors, not the silver
 # jetpacks, background-like shoes, or air covered by rough outline seeds.
 rgb=frame.astype(float);r,g,b=rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
 colored=((r>g*1.045)&(g>b*1.12))|((b>r*1.5)&(g>r*1.4)&(b<g*1.6))|((r>190)&(g>70)&(r>b*1.7))
 mask[inner&colored]=cv2.GC_FGD
 for x,y,radius in [(502,395,18),(502,300,14),(475,500,7),(541,494,7),(403,349,5),(567,349,5),(727,348,14),(729,445,9),(777,473,7),(726,252,10),(718,577,4),(826,574,4),(443,403,4),(552,404,4),(678,373,6),(777,378,4)]:
  cv2.circle(mask,(x,y),radius,int(cv2.GC_FGD),-1)
 # Sky and background openings inside loose flame polygons must stay optional.
 cv2.grabCut(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),mask,None,np.zeros((1,65)),np.zeros((1,65)),4,cv2.GC_INIT_WITH_MASK)
 fg=((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)).astype(np.uint8)
 return fg

def saveim(name,im):Image.fromarray(im).save(OUT/name)

def main():
 b=read_frames(ROOT/'IYTYT.mp4',2888,1)[0];m=grab(b)
 saveim('mask-2888.png',m*255)
 overlay=b.copy();overlay[m==0]=(overlay[m==0]*.3).astype(np.uint8);saveim('foreground-mask-check.png',overlay)
 check=np.full_like(b,160);check[m>0]=b[m>0];saveim('foreground-isolation.png',check)
 np.save(OUT/'mask-2888.npy',m)
 print('Mask prepared',flush=True)
if __name__=='__main__':main()
