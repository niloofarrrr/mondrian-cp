"""Verify interval-cell selection and local Psi bounds on design-only samples."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import conformal_shield as cs
rng=np.random.default_rng(610951)
for _ in range(1000):
    grid=np.cumsum(rng.uniform(.01,1.,20))-5.
    center=rng.uniform(grid[0]-1,grid[-1]+1);radius=rng.uniform(0,2)
    lo=np.clip(center-radius,grid[0],grid[-1]);hi=np.clip(center+radius,grid[0],grid[-1])
    expected=np.flatnonzero((grid[:-1]<=hi)&(grid[1:]>=lo))
    np.testing.assert_array_equal(cs.CBVFTable._tube_axis_cells(grid,center,radius),expected)
root=Path(__file__).resolve().parents[1]
t=cs.CBVFTable.from_npz(next((root/'results_intersample_v5/equation_correct_gamma2_seed610502/cbvf_cache').glob('*.npz')),max_abs_table=10,max_abs_grad=200)
t.enable_continuous_time_interpolation(True);model=cs.DynamicsSpec(.6,.8,-.2)
def psi(state,time,u):
    b,g,bt=t.value_grad_dt(state,time)
    v=.2+.4*u[0]
    return bt+2*b+v*(g[0]*np.cos(state[2])+g[1]*np.sin(state[2]))+.8*u[1]*g[2]
checks=0
for _ in range(160):
    center=np.r_[rng.uniform(-1.1,1.1,2),rng.uniform(-np.pi,np.pi)]
    start=rng.uniform(-16.9,-5.1);dt=rng.choice([.001,.002,.005]);radius=.6*dt
    row=t.certified_local_psi_lipschitz(center,start,start+dt,radius,radius,model,2.)
    u=rng.choice([-1.,1.],2);point=center+rng.uniform(-.45,.45,3)*radius;time=start+.5*dt
    for axis,key in enumerate(['x','y','theta','time']):
        h=1e-7;left=point.copy();right=point.copy();tl=tr=time
        if axis<3:left[axis]-=h;right[axis]+=h
        else:tl-=h;tr+=h
        derivative=abs((psi(right,tr,u)-psi(left,tl,u))/(2*h))
        assert derivative<=row['component_bounds'][key]+1e-5,(derivative,row,key)
        checks+=1
payload={'passed':True,'exact_cell_selection_cases':1000,'sampled_psi_derivatives_bounded':checks,'note':'Sampling supplements the Cartesian cell-enclosure argument; it is not a continuous-domain proof.'}
out=root/'results_intersample_v5/tight_cell_certificate';out.mkdir(exist_ok=True)
(out/'verification.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload))
