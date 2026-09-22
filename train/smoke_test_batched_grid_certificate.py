"""Compare every batched bound component's final result to scalar certificates."""
from pathlib import Path
import sys
import time
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import conformal_shield as cs
root=Path(__file__).resolve().parents[1]
table=cs.CBVFTable.from_npz(next((root/'results_intersample_v5/equation_correct_gamma2_seed610502/cbvf_cache').glob('*.npz')),max_abs_table=10,max_abs_grad=200)
model=cs.DynamicsSpec(.6,.8,-.2);rng=np.random.default_rng(610950)
indices=[tuple(rng.integers([31,31,25])) for _ in range(120)]+[(0,0,0),(30,30,24),(0,30,24),(15,15,12)]
maximum_error=0.;batch_seconds=0.;scalar_seconds=0.
for start,end,dt in [(-17.,-16.999,.001),(-10.101,-10.096,.005),(-5.1,-5.095,.005),(-.5,0.,.5)]:
    begin=time.perf_counter();bound=table.certified_grid_psi_lipschitz(start,end,.6*dt,.6*dt,model,2.);batch_seconds+=time.perf_counter()-begin
    begin=time.perf_counter()
    for index in indices:
        state=np.array([g[i] for g,i in zip((table.x_grid,table.y_grid,table.th_grid),index)])
        scalar=table.certified_local_psi_lipschitz(state,start,end,.6*dt,.6*dt,model,2.)['L_Psi_j']
        error=abs(bound[index]-scalar);maximum_error=max(maximum_error,error)
        assert bound[index]==scalar,(index,bound[index],scalar)
    scalar_seconds+=time.perf_counter()-begin
payload={'passed':True,'comparisons':len(indices)*4,'bitwise_equal':True,'maximum_error':maximum_error,'batch_all_nodes_seconds':batch_seconds,'scalar_sample_seconds':scalar_seconds}
out=root/'results_intersample_v5/batched_certificate';out.mkdir(exist_ok=True)
(out/'verification.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload),flush=True)
