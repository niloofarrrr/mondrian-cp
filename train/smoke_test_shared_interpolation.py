"""Check shared-stencil interpolation against the independent scalar path."""
import sys
from pathlib import Path
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import conformal_shield as cs

rng=np.random.default_rng(617001)
x=np.linspace(-2,2,9); y=np.linspace(-3,3,11)
h=np.linspace(-np.pi,np.pi,13,endpoint=False); t=np.linspace(-4,0,7)
table=cs.CBVFTable(x,y,h,t,rng.uniform(-.5,.5,(9,11,13,7)),max_abs_table=10,max_abs_grad=200)
table.enable_continuous_time_interpolation(True)
for grid in (x,y,h,t):
    for value in np.r_[rng.uniform(-10,10,1000),grid,-np.inf,np.inf]:
        clipped=float(np.clip(value,grid[0],grid[-1]))
        j=max(0,min(int(np.searchsorted(grid,clipped,side='right')-1),len(grid)-2))
        expected_weight=(clipped-grid[j])/(grid[j+1]-grid[j])
        assert table._cell_and_weight(grid,value)==(j,expected_weight)
points=[(rng.uniform([-3,-4,-4*np.pi],[3,4,4*np.pi]),rng.uniform(-5,1)) for _ in range(1000)]
points += [(np.array([xx,yy,hh]),tt) for xx in (-2,2) for yy in (-3,3) for hh in (-np.pi,np.pi) for tt in (-4,0)]
start=time.perf_counter()
expected=[np.array([table._interp_space_time(f,s,tt) for f in (table.B_table,table.gx,table.gy,table.gth,table.gt)]) for s,tt in points]
scalar_seconds=time.perf_counter()-start
start=time.perf_counter()
for (s,tt),want in zip(points,expected):
    b,g,bt=table.value_grad_dt(s,tt)
    np.testing.assert_array_equal(np.r_[b,g,bt],want)
shared_seconds=time.perf_counter()-start
print('Shared interpolation: exact equality at',len(points),'points; scalar/shared seconds:',scalar_seconds,shared_seconds)
