"""One design-only reference CP trajectory to diagnose task progress."""
from pathlib import Path
import sys,json,math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import conformal_shield as cs
from train.train_sac_lag import _reference_warmup_action
root=Path(__file__).resolve().parents[1];source=root/'results_intersample_v5/gamma15_beta054_reverse010_dt001_x130_seed610508'
cfg=json.loads((source/'batch_configuration.json').read_text())['configuration']
saved=json.loads((source/'pretraining_mondrian_calibration.json').read_text())['mondrian_calibration'];regions=saved['regions']
partition=cs.MondrianPartition(tuple(saved['clearance_edges']),(0.,0.),.7)
cal=cs.MondrianCalibration(partition,tuple(tuple(r['base_regions']) for r in regions),tuple(saved['base_to_effective']),tuple(r['xi_hat_off'] for r in regions),tuple(r['delta_m'] for r in regions),tuple(r['n_scores'] for r in regions),saved['delta_traj'],saved['epsilon_grid'],saved['promotion_rounds'])
table=cs.CBVFTable.from_npz(next((source/'cbvf_cache').glob('*.npz')),max_abs_table=10,max_abs_grad=200);table.enable_continuous_time_interpolation(True)
variant=cs.DynamicsVariant('design',cs.DynamicsSpec(.6,.8,-.2),cs.DynamicsSpec(.6,.54,-.1));dt=.001
world=cs.World(goal=(1.5,1.5));env=cs.DubinsCBVFEnv(world,cs.Interval(-1,1),variant,dt,12000);env.set_state(np.array([-1.3,-3,np.pi/4]))
rows=[];minimum=100.;infeasible=0
for i in range(env.horizon):
 s=env.state.copy();t=-17+i*dt
 buffer,_=cal.applied_buffer(s,.6*dt)
 L=table.certified_local_psi_lipschitz(s,t,t+dt,.6*dt,.54*dt,variant.model,1.5)['L_Psi_j']
 rhs=buffer+L*(1+math.hypot(.6,.54))*dt
 projection=cs.solve_cbvf_projection_exact(s,t,_reference_warmup_action(s,.8,world.goal),table,env,1.5,rhs)
 infeasible+=not projection.feasible;s,done,info=env.step(projection.u);minimum=min(minimum,env.safety_margin_value(s))
 if i%100==0 or done: rows.append([i*dt,*s,*projection.u,buffer,rhs])
 if done:break
out=root/'results_intersample_v5/cp_progress_diagnosis';out.mkdir(exist_ok=False)
np.savetxt(out/'trajectory.csv',rows,delimiter=',',header='elapsed,x,y,theta,u_speed,u_yaw,buffer,rhs',comments='')
result={'design_only':True,'source':str(source),'steps':i+1,'final_state':s.tolist(),'distance_to_goal':cs.goal_distance(world,s),'goal':bool(info['goal']),'unsafe':bool(info['unsafe']),'minimum_physical_clearance':minimum,'infeasible_steps':int(infeasible)}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
