"""Reject sparse target buffers and verify two independent identical HJ solves."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import conformal_shield as cs
from odp.Grid import Grid
from solver_cbvf import HJSolver

root=Path(__file__).resolve().parents[1]
out=root/'results_intersample_v4/solver_reproducibility'
out.mkdir(parents=True,exist_ok=True)
world=cs.World()
grid=Grid(np.array([-4.,-4.,-np.pi]),np.array([4.,4.,np.pi]),3,np.array([31,31,25]),[2])
target=cs.build_world_target_on_odp_grid(world,grid,'signed_distance',1.,0.)
assert target.shape==(31,31,25) and target.flags.c_contiguous
np.testing.assert_array_equal(target,np.broadcast_to(target[:,:,:1],target.shape))
try:
    HJSolver(cs.ModelDubinsCBVFODP(.6,-.2,.8,cs.Interval(-1,1)),grid,
             target[:,:,:1],np.array([0.,.1]),{'TargetSetMode':'maxCBF'})
except ValueError as error:
    assert 'Target shape' in str(error)
else:
    raise AssertionError('Sparse target was not rejected')
tables=[]
for i in range(2):
    table,_,path=cs.prepare_cbvf_table(cache_dir=out/str(i),world=world,
        control_bounds=cs.Interval(-1,1),model_spec=cs.DynamicsSpec(.6,.8,-.2),
        gamma=.25,dt=.001,horizon=17000,cbvf_dt=.1,nx=31,ny=31,nth=25,
        accuracy='low',recompute=True,odp_root=str(root),solver_file=str(root/'solver_cbvf.py'),
        target_shape='signed_distance',target_clip=1.,target_scale=0.,max_abs_table=10.,
        max_abs_grad=200.,max_abs_solver_value=50.,allow_coarse_fallback=False,
        max_est_runtime_gb=3.,use_time_invariant_cbvf=False)
    np.testing.assert_array_equal(table.B_table[...,-1],target)
    tables.append(table.B_table)
np.testing.assert_array_equal(tables[0],tables[1])
payload={'passed':True,'target_shape':list(target.shape),'table_shape':list(tables[0].shape),
         'identical_fresh_solves':True,'B_array_sha256':hashlib.sha256(tables[0].tobytes()).hexdigest()}
(out/'verification.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload),flush=True)
