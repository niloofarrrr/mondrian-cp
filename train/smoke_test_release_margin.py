import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import conformal_shield as cs
x=np.linspace(-1,1,11); y=x.copy(); th=np.linspace(-np.pi,np.pi,9,endpoint=False); t=np.linspace(-.1,0,6)
X,Y,H,T=np.meshgrid(x,y,th,t,indexing='ij')
b=cs.CBVFTable(x,y,th,t,X,max_abs_table=10.)
v=cs.DynamicsVariant('test',cs.DynamicsSpec(.6,.8,0),cs.DynamicsSpec(.6,.8,0))
w=cs.World(obstacle_center=(10.,10.))
counts=[]
for release in (0.,.05):
 e=cs.DubinsCBVFEnv(w,cs.Interval(-1,1),v,.02,5)
 r=cs.rollout(e,b,lambda s:np.array([1.,0.]),t,np.array([.02,0.,0.]),'cbvf',.25,0.,cs.ShieldConfig(.03,.03,release_margin=release))
 counts.append(r['active_filter_steps'])
assert counts == [1.,5.], counts
print('Configured release-margin rollout regression: PASS',counts)
