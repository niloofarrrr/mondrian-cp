"""Verify optimized box clips preserve controls and complete QP solutions."""
from pathlib import Path
import importlib.util,sys,json,time
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import conformal_shield as cs
spec=importlib.util.spec_from_file_location('prior_clip_cs',root/'results_intersample_v5/clip_optimization/conformal_shield_before.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
rng=np.random.default_rng(610952);model=cs.DynamicsSpec(.6,.8,-.2)
for u in list(rng.normal(size=(1000,2)))+[np.array([np.inf,-np.inf]),np.array([-0.,0.])]:
 assert cs.DubinsCBVFEnv._physical_controls(model,u)==old.DubinsCBVFEnv._physical_controls(model,u)
t=cs.CBVFTable.from_npz(next((root/'results_intersample_v5/gamma15_beta054_reverse010_dt001_x130_seed610508/cbvf_cache').glob('*.npz')),max_abs_table=10,max_abs_grad=200);t.enable_continuous_time_interpolation(True)
env=cs.DubinsCBVFEnv(cs.World(),cs.Interval(-1,1),cs.DynamicsVariant('design',model,cs.DynamicsSpec(.6,.54,-.1)),.001,12000)
queries=[(np.r_[rng.uniform(-1.5,1.5,2),rng.uniform(-np.pi,np.pi)],rng.uniform(-17,-5),rng.uniform(-2,2,2),rng.uniform(0,.2)) for _ in range(600)]
results=[];seconds=[]
for module in [old,cs]:
 begin=time.perf_counter();results.append([module.solve_cbvf_projection_exact(s,time,u,t,env,1.5,xi) for s,time,u,xi in queries]);seconds.append(time.perf_counter()-begin)
for left,right in zip(*results):
 for name in vars(left):np.testing.assert_array_equal(getattr(left,name),getattr(right,name))
payload={'passed':True,'physical_control_cases':1002,'complete_qp_solutions_identical':600,'old_new_seconds':seconds}
(root/'results_intersample_v5/clip_optimization/verification.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload))
