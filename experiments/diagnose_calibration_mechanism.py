"""Retrospective mechanism/chain-rule audit; never a new final evaluation."""
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import conformal_shield as cs


def main():
    source = ROOT/'results_intersample_v5/persistent_workflow/studies/study_003/final_suite/runs/hard_cp_seed_71000001'
    out = ROOT/'results_mechanism_v6/diagnosis'
    out.mkdir(parents=True, exist_ok=True)
    result = json.loads((source/'training_and_evaluation_results.json').read_text())
    saved = json.loads((source/'final_frozen_policy_mondrian_calibration.json').read_text())
    table = cs.CBVFTable.from_npz(saved['cbvf_path'], max_abs_table=10, max_abs_grad=200)
    table.enable_continuous_time_interpolation(True)
    variant = cs.DynamicsVariant('retrospective', cs.DynamicsSpec(.6,.8,-.2), cs.DynamicsSpec(.6,.43,-.15))
    env = cs.DubinsCBVFEnv(cs.World(goal=(0,1.8)),cs.Interval(-1,1),variant,.002,6000)
    raw = saved['mondrian_calibration']; regions = raw['regions']
    cal = cs.MondrianCalibration(cs.MondrianPartition(tuple(raw['clearance_edges']),(0.,0.),.7),
        tuple(tuple(r['base_regions']) for r in regions),tuple(raw['base_to_effective']),
        tuple(r['xi_hat_off'] for r in regions),tuple(r['delta_m'] for r in regions),
        tuple(r['n_scores'] for r in regions),raw['delta_traj'],raw['epsilon_grid'],raw['promotion_rounds'])
    rows = []; summaries = {}
    for method in ('nominal','cbvf','cp'):
        trajectory = result['final_evaluation'][method]['representative_trajectory']
        for j,(state,nominal,applied) in enumerate(zip(trajectory['states'],trajectory['nominal_actions'],trajectory['applied_actions'])):
            state = np.array(state); nominal = np.array(nominal); applied = np.array(applied)
            h = float(np.linalg.norm(state[:2])-.7)
            if h > .35 or j % 5: continue
            t = -17 + j*.002; B,g,bt = table.value_grad_dt(state,t)
            terms = cs.model_constraint_terms(state,t,table,env,1.5)
            def f(spec,u):
                v=spec.v_min+(u[0]+1)*(spec.v-spec.v_min)/2
                return np.array([v*np.cos(state[2]),v*np.sin(state[2]),spec.beta_u*u[1]])
            L = table.certified_local_psi_lipschitz(state,t,t+.002,.6*.002,.43*.002,variant.model,1.5)['L_Psi_j']
            epsilon=L*(1+math.hypot(.6,.43))*.002
            xi,_=cal.applied_buffer(state,math.hypot(.6,.43)*.002)
            # Common intersample term for a fair same-state projection comparison.
            common_L=table.certified_local_psi_lipschitz(state,t,t+.002,.6*.002,.8*.002,variant.model,1.5)['L_Psi_j']
            common_eps=common_L*2*.002
            uncal=cs.solve_cbvf_projection_exact(state,t,nominal,table,env,1.5,common_eps)
            calibrated=cs.solve_cbvf_projection_exact(state,t,nominal,table,env,1.5,common_eps+xi)
            eps=1e-6; exact_g=[]
            for axis in range(3):
                e=np.zeros(3);e[axis]=eps
                exact_g.append((table.value_grad_dt(state+e,t)[0]-table.value_grad_dt(state-e,t)[0])/(2*eps))
            exact_g=np.array(exact_g)
            exact_bt=(table.value_grad_dt(state,t+eps)[0]-table.value_grad_dt(state,t-eps)[0])/(2*eps)
            psi_model=bt+g@f(variant.model,applied)+1.5*B
            psi_true=bt+g@f(variant.truth,applied)+1.5*B
            psi_chain=exact_bt+exact_g@f(variant.truth,applied)+1.5*B
            rows.append(dict(trajectory=method,step=j,time=t,x=state[0],y=state[1],theta=state[2],
                h=h,B=B,B_minus_h=B-h,nominal_0=nominal[0],nominal_1=nominal[1],applied_0=applied[0],applied_1=applied[1],
                psi_model_applied=psi_model,psi_true_field_applied=psi_true,psi_true_chain_rule=psi_chain,
                psi_nominal=terms.base_without_control+terms.a_u@nominal,gamma_B=1.5*B,B_t=bt,
                xi=xi,L_Psi=L,epsilon_inter=epsilon,common_epsilon=common_eps,
                max_psi=terms.base_without_control+np.abs(terms.a_u).sum(),
                original_uncal_feasibility=terms.base_without_control+np.abs(terms.a_u).sum(),
                common_uncal_feasibility=uncal.max_achievable_lhs-common_eps,
                common_cp_feasibility=calibrated.max_achievable_lhs-common_eps-xi,
                common_uncal_action_0=uncal.u[0],common_uncal_action_1=uncal.u[1],
                common_cp_action_0=calibrated.u[0],common_cp_action_1=calibrated.u[1],
                gradient_error_norm=float(np.linalg.norm(g-exact_g)),chain_rule_error=psi_true-psi_chain))
        rr=[r for r in rows if r['trajectory']==method]
        summaries[method]=dict(samples=len(rr),unsafe_rate=result['final_evaluation'][method]['unsafe_rate'],
            goal_rate=result['final_evaluation'][method]['goal_rate'],
            min_h=min(r['h'] for r in rr),min_B=min(r['B'] for r in rr),
            min_B_minus_h=min(r['B_minus_h'] for r in rr),max_B_minus_h=max(r['B_minus_h'] for r in rr),
            min_model_psi=min(r['psi_model_applied'] for r in rr),
            min_true_field_psi=min(r['psi_true_field_applied'] for r in rr),
            model_safe_true_field_violations=sum(r['psi_model_applied']>=-1e-8 and r['psi_true_field_applied'] < -1e-8 for r in rr),
            max_gradient_error=max(r['gradient_error_norm'] for r in rr),
            max_chain_rule_error=max(abs(r['chain_rule_error']) for r in rr),
            field_safe_chain_rule_violations=sum(r['psi_true_field_applied']>=0 and r['psi_true_chain_rule'] < -1e-8 for r in rr))
    with (out/'same_state_derivative_diagnostics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(source=str(source),purpose='retrospective diagnosis only, not parameter selection or a new final study',
                 sampled_every_n_intervals=5,clearance_upper=.35,matched_frozen_actor=True,summaries=summaries)
    serialized=json.dumps(summary,indent=2,default=lambda x:x.item())
    (out/'mechanism_summary.json').write_text(serialized+'\n')
    print(serialized,flush=True)


if __name__=='__main__':main()
