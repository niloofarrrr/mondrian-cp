#!/usr/bin/env python3
"""Freeze a qualified three-condition design before any reserved final run."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import run_full_suite as batch
from audit_pilot_certificates import audit_certificates
from effective_training_parameters import effective_parameters
ROOT=Path(__file__).resolve().parents[1]
SEEDS=[11000001,21000001,31000001,41000001,51000001]
CORE=['reference_waypoint_x','reference_waypoint_y','dt','horizon','gamma','initial_x','initial_y','initial_theta','goal_x','goal_y','start_jitter_xy','start_jitter_theta','cbvf_activate_margin','cp_activate_margin','shield_release_margin','mondrian_clearance_edges','n_calib','delta','cbvf_nx','cbvf_ny','cbvf_nth','cbvf_dt','cbvf_terminal_guard','intersample_protection','integration_substeps','residual_reference_scale']
DYNAMICS=['speed','speed_min','beta_u','cbvf_model_speed','cbvf_model_speed_min','cbvf_model_beta_u']
CORE += ['reference_orbit_trigger_radius', 'reference_orbit_radius', 'reference_orbit_gain']
def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='append',required=True,help='condition:path; repeat for aligned, easy, and hard')
    p.add_argument('--output',default='results_intersample_v5/frozen_study')
    p.add_argument('--final-seeds', type=int, nargs=5, default=SEEDS)
    a=p.parse_args()
    seeds=list(a.final_seeds)
    if len(set(seeds)) != 5 or min(seeds) < 11000001 or min(b-a for a,b in zip(sorted(seeds),sorted(seeds)[1:])) < 500000:
        raise ValueError('Requires five distinct fresh final seed blocks separated by at least 500000')
    for prior_path in (ROOT/'results_intersample_v5').rglob('FREEZE.json'):
        if 'source_snapshot' in prior_path.parts:continue
        prior=json.loads(prior_path.read_text())
        if any(abs(x-y)<500000 for x in seeds for y in prior.get('reserved_final_seeds',[])):
            raise ValueError('Final seed block overlaps a previous immutable freeze: '+str(prior_path))
    configs={};evidence=[];core=None
    for item in a.pilot:
        condition,name=item.split(':',1)
        if condition not in ['aligned','easy','hard']:raise ValueError(condition)
        folder=Path(name).resolve();record=json.loads((folder/'batch_configuration.json').read_text());cfg=record['configuration']
        certificate=audit_certificates(folder)
        if not certificate['passed']:raise ValueError(certificate['failures'])
        interval_audit=json.loads((folder/'runtime_interval_audit.json').read_text())
        if not interval_audit['passed']:raise ValueError('Runtime interval reconstruction failed: '+str(folder))
        for source in ['conformal_shield.py','solver_cbvf.py','train/train_sac_lag.py','redexp/envs/conformal_dubins_3d_env.py']:
            if record['source_hashes'][source]!=hashlib.sha256((ROOT/source).read_bytes()).hexdigest():
                raise ValueError('Simulation source changed since qualification: '+source)
        if not 620000<=cfg['seed']<=629999 or cfg['max_steps']<20000:raise ValueError('Requires a learned qualification pilot, not a geometry screen')
        if cfg['training_shield_method']!='cp':raise ValueError('Qualification pilot must use CP training')
        current={k:cfg[k] for k in CORE}
        if core is not None and core!=current:raise ValueError('Qualification simulation settings differ')
        core=current
        result=json.loads((folder/'training_and_evaluation_results.json').read_text())
        for phase in ['pretraining','final_frozen_policy']:
            audit=json.loads((folder/(phase+'_mondrian_calibration.json')).read_text())['offline_cp_qp_feasibility_audit']
            assert audit['n_infeasible_grid_nodes']==audit['n_uncalibrated_infeasible_grid_nodes']==0
            assert min(audit['min_feasibility_margin'],audit['min_uncalibrated_feasibility_margin'])>0
        final=result['final_evaluation'];cp=final['cp']
        assert cp['goal_rate']>=.90 and cp['unsafe_rate']==0
        assert cp['infeasible_total']==cp['rail_fallback_total']==cp['minimal_projection_failure_total']==0
        assert cp['min_active_feasibility_margin']>0
        if condition=='hard':assert max(final[m]['unsafe_rate'] for m in ['nominal','cbvf'])>=.10
        coverage=json.loads((folder/'heldout_reference_coverage_fresh.json').read_text())
        assert coverage['episodes']>=100 and coverage['fresh_calibration'] and coverage['score_population_matches_calibration']
        assert coverage['disjointness']['overlap_pairs']==0 and coverage['empirical_simultaneous_coverage']>=.95
        if condition in configs and any(configs[condition][k]!=cfg[k] for k in DYNAMICS):
            raise ValueError('Qualification dynamics differ within condition '+condition)
        configs[condition]=cfg
        evidence.append({'condition':condition,'path':str(folder),'result_sha256':hashlib.sha256((folder/'training_and_evaluation_results.json').read_bytes()).hexdigest(),'coverage_sha256':hashlib.sha256((folder/'heldout_reference_coverage_fresh.json').read_bytes()).hexdigest()})
    if set(configs)!=set(['aligned','easy','hard']):raise ValueError('All three conditions must qualify')
    out=(ROOT/a.output).resolve()
    if out.exists():raise FileExistsError('A freeze is immutable: '+str(out))
    common=dict(configs['hard'])
    for key in DYNAMICS+['seed','training_shield_method','checkpoint_dir','results_json','eval_jsonl','action_log_jsonl','qp_diagnostics_jsonl','plot_dir','cache_dir','odp_root','solver_file','feasibility_only','calibration_manifest_in']:
        common.pop(key,None)
    common.update(max_steps=250000,start_training=25000,reference_proposal_steps=75000,eval_interval=25000,save_interval=25000,eval_episodes=100,periodic_eval_episodes=5,recompute_cbvf=True,reuse_cbvf=False,reuse_zero_offline_audit=False)
    manifest={'suite_name':'mondrian_cbvf_equation_validated_v5','results_dir':str(out/'final_suite'),'python':'/home/mars/miniconda3/envs/odp/bin/python3.8','fixed_seeds':seeds,'methods':['nominal','cbvf','cp'],'conditions':{name:{k:cfg[k] for k in DYNAMICS} for name,cfg in configs.items()},'common':common,'isolated_cbvf_cache':True,'heldout_coverage_episodes':100}
    effective=effective_parameters(manifest,batch)
    sources=[ROOT/'conformal_shield.py',ROOT/'solver_cbvf.py']
    for name in ['train','experiments','redexp','jaxrl5','odp']:
        sources.extend((ROOT/name).rglob('*.py'))
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources if f.is_file()}
    freeze={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':batch.config_hash(manifest),'source_hashes':hashes,'qualification_evidence':evidence,'reserved_final_seeds':seeds,'seed_offsets':{'pretraining_calibration':20000,'final_calibration':30000,'final_performance':340000,'heldout_reference':370000},'final_data_used_for_design':False}
    out.mkdir(parents=True)
    freeze['effective_parameters_sha256']=batch.config_hash(effective)
    (out/'effective_parameters.json').write_text(json.dumps(effective,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
    for name in hashes:
        destination=out/'source_snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,destination)
    print(out)
if __name__=='__main__':main()
