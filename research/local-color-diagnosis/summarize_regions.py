"""Matched flat-interior color evidence in visually selected native regions. No edits."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
D=Path(__file__).resolve().parent
regions={
722:[('hair','Woman brown hair',[420,80,454,125]),('car_pale','Pale upper car paint',[840,458,970,477]),('car_pink','Pink car paint',[355,560,425,592]),('car_purple','Purple car paint',[680,607,780,626]),('car_teal','Teal car paint',[595,548,690,575]),('sky','Night sky',[540,70,670,180])],
1083:[('hair','Woman brown hair',[917,202,967,228]),('tracksuit','Woman blue tracksuit',[882,450,909,533]),('car_pink','Pink car paint',[646,467,704,494]),('car_teal','Teal car paint',[50,480,245,507]),('car_pale','Pale car paint',[350,387,495,405]),('fur','Sloth brown fur',[259,296,276,332])],
1444:[('hair','Woman brown hair',[233,180,289,197]),('tracksuit','Woman blue tracksuit',[228,268,269,301]),('car_pale','Pale trunk paint',[505,373,580,408]),('car_pink','Pink lower car paint',[536,524,685,539]),('truck_gray','Gray flat truck panel',[69,115,120,250]),('road_blue','Blue asphalt',[930,535,1000,565]),('fur','Sloth back fur',[535,279,567,320])],
1805:[('foliage','Near right tree green canopy',[1160,42,1220,115]),('foliage_mid','Mid-distance tree canopy',[356,259,390,283]),('blue_window','Flat blue shop window',[138,321,177,363]),('car_pale','Pale trunk paint',[578,374,611,390]),('sky','Night sky',[620,25,701,65]),('road_blue','Blue asphalt',[245,515,400,595])],
2166:[('sky_left','Blue sky left',[100,15,270,70]),('sky_center','Blue sky center',[654,15,807,73]),('sky_right','Blue sky right',[908,34,1140,89]),('skyline_dark','Dark skyline building',[383,104,399,122]),('river_dark','Dark river',[260,202,325,226]),('river_mid','Dark river right',[963,211,1075,236])],
2527:[('tracksuit','Woman cyan tracksuit',[1174,405,1206,447]),('wall','Cream apartment wall',[210,35,330,90]),('couch','Cream couch upholstery',[685,336,773,385]),('fur','Sloth brown body fur',[500,418,539,442]),('tablet','Gray tablet',[505,330,555,375]),('table_shadow','Black underside of table',[390,634,698,651]),('wood','Warm wooden tabletop',[330,531,494,547])],
2888:[('sky','Daytime sky',[915,40,1175,250]),('fur','Sloth brown body fur',[474,379,515,425]),('tracksuit','Woman blue tracksuit',[726,413,744,452]),('water','Pale blue river',[943,417,988,441]),('building_gray','Broad gray building wall',[232,413,276,430])],
3240:[('hair','Woman brown hair',[708,290,732,317]),('tracksuit','Woman blue tracksuit',[704,359,730,377]),('fur','Sloth brown fur',[540,400,554,433]),('steps','Beige stone steps',[378,624,671,639]),('stone','Pale stone column base',[193,493,270,536]),('door_wood','Dark brown door panel',[890,217,902,237]),('ceiling','Cream ceiling panel',[598,202,727,219])]
}
observations={
722:'Pale car paint darkens while neighboring pink/purple paint loses magenta. Hair shifts more yellow/orange. Teal panel changes much less. Texture/reflection streaks excluded.',
1083:'Blue tracksuit and pale car paint darken while brown hair/fur warm. Adjacent teal/pink car regions respond differently. Strong evidence of color-conditioned grade residual.',
1444:'Pale trunk and blue tracksuit darken while pink bumper gets less magenta and brown hair shifts yellow. Truck panel and road are more stable than those material colors.',
1805:'Green foliage loses green brightness much more than blue asphalt/sky. Small distant character shapes are too small for trustworthy interior sampling here.',
2166:'Upper sky/river shows broad darkening and loss of red/green. Skyline and small rooftops also redraw, so pixel differences in detailed city are not pure color. Largest all-frame color residual among joins.',
2527:'Cyan tracksuit saturates blue; gray tablet/wall/couch respond differently. Black underside of coffee table lifts to dark gray. Source-exact 0/255 preservation makes a symmetric midpoint grade unable to lift the zero side.',
2888:'Sloth fur warms while daytime blue sky darkens slightly. Background changes relative scale at this unresolved geometry join; only flat correspondence-consistent surfaces retained. Fine city windows excluded.',
3240:'Brown hair/wood shift toward yellow/orange while cyan tracksuit darkens. Beige stone and broad cream ceiling remain much closer. Thin architectural contours/text excluded.'}
out=[]
for cut,rs in regions.items():
 z=np.load(D/str(cut)/'matched-flat-samples.npz'); xy=z['left_xy_native']; l=z['left_rgb'];r=z['right_rgb']
 im=Image.open(D/str(cut)/'left.png').convert('RGB');draw=ImageDraw.Draw(im);rr=[]
 for i,(name,material,rect) in enumerate(rs,1):
  x1,y1,x2,y2=rect;m=(xy[:,0]>=x1)&(xy[:,0]<x2)&(xy[:,1]>=y1)&(xy[:,1]<y2)
  delta=r[m]-l[m];n=int(m.sum());rec={'id':name,'material':material,'left_rect_xyxy':rect,'accepted_matched_samples':n}
  if n:
   rec.update(left_rgb_median=np.median(l[m],axis=0).tolist(),right_rgb_median=np.median(r[m],axis=0).tolist(),signed_rgb_mean_right_minus_left=delta.mean(0).tolist(),signed_rgb_median_right_minus_left=np.median(delta,axis=0).tolist(),residual_mad_about_signed_median=np.median(abs(delta-np.median(delta,axis=0)),axis=0).tolist(),rgb_mae=float(abs(delta).mean()))
  rec['confidence']='good for signed regional bias; residual texture and flow uncertainty remain' if n>=30 else 'limited sample count; corroborating only'
  rr.append(rec);draw.rectangle(rect,outline=(255,60,200),width=2);draw.text((x1,max(0,y1-12)),str(i)+':'+name,fill=(255,255,255),stroke_width=1,stroke_fill=(0,0,0))
  print(cut,name,n,'delta',np.round(delta.mean(0),2) if n else 'NONE')
 im.save(D/str(cut)/'material-regions-annotated.png')
 out.append({'frame':cut,'visual_observation':observations[cut],'regions':rr})
json.dump({'measurement_only':True,'method':'Visually selected native rectangles. Samples use existing correspondence-consistent flat matches with five-native-pixel contour exclusion, not fixed-rectangle pixel subtraction. RGB units are 8-bit; signed delta is incoming minus outgoing. No models fitted here.','cuts':out},open(D/'visual-other-joins.json','w'),indent=2)
