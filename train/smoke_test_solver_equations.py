"""Analytic CBVF tests that detect discount-sign and half-advection defects."""
from pathlib import Path
import sys
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from odp.Grid import Grid
from solver_cbvf import HJSolver
class Translation:
    def opt_ctrl(self,*args): return (0.,0.,0.)
    def opt_dstb(self,*args): return (0.,0.,0.)
    def dynamics(self,*args): return (-1.,0.,0.)
grid=Grid(np.array([-2.,-1.,-1.]),np.array([2.,1.,1.]),3,np.array([21,5,5]),[])
shape=tuple(grid.pts_each_dim)
def solve(target,gamma,T):
    return HJSolver(Translation(),grid,np.asarray(target,np.float32),np.linspace(0,T,11),
        {'TargetSetMode':'maxCBF','cbf_gamma':gamma,'max_abs_value':100.,'value_clip_lo':-100.,'value_clip_hi':100.},
        saveAllTimeSteps=True,accuracy='low')[...,0]
positive=solve(np.full(shape,.5),.25,1.)
np.testing.assert_allclose(positive,.5,rtol=0,atol=1e-7)
negative=solve(np.full(shape,-.2),.25,1.)
np.testing.assert_allclose(negative,-.2*np.exp(.25),rtol=1e-6,atol=1e-7)
affine=np.broadcast_to(grid.vs[0],shape).copy()
translated=solve(affine,0.,.2)
error=float(np.max(np.abs(translated[7:14]- (affine[7:14]-.2))))
assert error<1e-5,error
out=Path(__file__).resolve().parents[1]/'results_intersample_v5/solver_equations'
out.mkdir(parents=True,exist_ok=True)
payload={'passed':True,'positive_stationary_value':float(positive[10,2,2]),'negative_reaction_value':float(negative[10,2,2]),'negative_reaction_expected':float(-.2*np.exp(.25)),'advection_max_error':error,'manuscript_equation':'U_s = max_u grad(U).f + gamma U; U <= l'}
(out/'verification.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload),flush=True)
