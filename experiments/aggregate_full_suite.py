#!/usr/bin/env python3
"""Aggregate the definitive 3x3x5 suite into publication artifacts."""
import argparse, csv, json, math
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import conformal_shield as cs

STATE = ROOT / "results" / "run_state.json"
OUT = ROOT / "results" / "aggregate"
MANIFEST = ROOT / "experiments" / "full_suite.json"
METHODS = ("nominal", "cbvf", "cp")
CONDITIONS = ("aligned", "easy", "hard")
COLORS = {"nominal":"#555555", "cbvf":"#2878b5", "cp":"#2ca25f"}
LABELS = {"nominal":"Nominal RL", "cbvf":"Uncalibrated CBVF", "cp":"Mondrian conformal CBVF"}

def save_figure(fig, stem):
    fig.savefig(OUT/(stem+".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT/(stem+".pdf"), bbox_inches="tight")
    plt.close(fig)

def heldout_coverage(d):
    path=Path(d["calibration_manifest"]).parent/"heldout_reference_coverage_fresh.json"
    payload=json.loads(path.read_text())
    if payload["episodes"] < 100 or not payload["fresh_calibration"]:
        raise RuntimeError(f"invalid held-out coverage artifact: {path}")
    if payload["disjointness"]["overlap_pairs"] != 0:
        raise RuntimeError(f"calibration/evaluation overlap in {path}")
    return float(payload["empirical_simultaneous_coverage"])

def oracle_cbvf_path(condition):
    manifest=json.loads(MANIFEST.read_text())
    common=manifest["common"]; dyn=manifest["conditions"][condition]
    world=cs.World(xmin=-4,xmax=4,ymin=-4,ymax=4,goal=(common.get('goal_x',2.),common.get('goal_y',2.)),goal_radius=.3,
        obstacle_center=(0,0),obstacle_radius=.5,robot_radius=.2,
        init_x_range=(-2,-2),init_y_range=(-2,-2),init_theta_range=(np.pi/4,np.pi/4))
    _,_,path=cs.prepare_cbvf_table(
        cache_dir=OUT/"oracle_tables"/condition,world=world,
        control_bounds=cs.Interval(-1,1),
        model_spec=cs.DynamicsSpec(v=dyn["speed"],v_min=dyn["speed_min"],beta_u=dyn["beta_u"]),
        gamma=common["gamma"],dt=common["dt"],
        horizon=int(math.ceil((common["horizon"]*common["dt"]+common["cbvf_terminal_guard"])/common["dt"])),
        cbvf_dt=common["cbvf_dt"],nx=common["cbvf_nx"],ny=common["cbvf_ny"],nth=common["cbvf_nth"],
        accuracy="low",recompute=True,odp_root=str(ROOT),solver_file=str(ROOT/"solver_cbvf.py"),
        target_shape=common["target_shape"],target_clip=1.0,target_scale=0.0,
        max_abs_table=10,max_abs_grad=200,max_abs_solver_value=50,
        allow_coarse_fallback=False,max_est_runtime_gb=3,use_time_invariant_cbvf=False)
    return path

def representative_seed():
    return int(json.loads(MANIFEST.read_text())["fixed_seeds"][0])

def load():
    state=json.loads(STATE.read_text())
    if not all(r["status"]=="completed" for r in state["runs"].values()):
        raise SystemExit("Suite is incomplete; aggregation refuses partial/selective results.")
    runs=[]
    for r in state["runs"].values():
        d=json.loads(Path(r["result_paths"]["result"]).read_text())
        method=r["configuration"]["training_shield_method"]
        # The result JSONs contain large trajectory/action payloads.  Retain
        # only states actually consumed by Figures 2 and 4; keeping every
        # duplicated trajectory from all 45 files can exceed laptop memory.
        for evaluation in d.get("evaluation_records", []):
          for values in evaluation.get("methods", {}).values():
            values.pop("representative_trajectory", None)
        for evaluated_method, values in d.get("final_evaluation", {}).items():
          keep_states = (
              (method == "cp" and evaluated_method == "cp")
              or (method == "cp" and r["seed"] == representative_seed())
          )
          trajectory = values.get("representative_trajectory")
          if keep_states and trajectory:
            values["representative_trajectory"] = {
                "states": trajectory.get("states", [])
            }
          else:
            values.pop("representative_trajectory", None)
        runs.append((r["condition"],method,r["seed"],d))
    return runs

def mean_sd(values):
    a=np.asarray(values,float); return float(a.mean()), float(a.std(ddof=1)) if len(a)>1 else 0.0

def primary(d, method): return d["final_evaluation"][method]

def figure1(runs):
    fig,ax=plt.subplots(2,3,figsize=(14,7),sharex="col",sharey="row")
    for col,c in enumerate(CONDITIONS):
      for m in METHODS:
        rr=[d for cc,mm,s,d in runs if cc==c and mm==m]
        curves=[]
        for d in rr:
          rec=d["evaluation_records"]
          bystep={int(x["step"]):x["methods"][m] for x in rec}
          curves.append(bystep)
        steps=sorted(set.intersection(*(set(x) for x in curves)))
        ret=np.array([[x[s]["return"] for s in steps] for x in curves])
        violations=np.array([[x[s]["total_safety_violations"] for s in steps] for x in curves])
        mu=ret.mean(0); ci=2.776*ret.std(0,ddof=1)/np.sqrt(len(ret))
        ax[0,col].plot(steps,mu,color=COLORS[m],label=LABELS[m]); ax[0,col].fill_between(steps,mu-ci,mu+ci,color=COLORS[m],alpha=.18)
        cum=np.cumsum(violations,axis=1); mu=cum.mean(0); ci=2.776*cum.std(0,ddof=1)/np.sqrt(len(cum))
        ax[1,col].plot(steps,mu,color=COLORS[m],label=LABELS[m]); ax[1,col].fill_between(steps,mu-ci,mu+ci,color=COLORS[m],alpha=.18)
      ax[0,col].set_title(c.title()); ax[1,col].set_xlabel("training steps")
      for a in ax[:,col]: a.grid(alpha=.25)
    ax[0,0].set_ylabel("episodic return"); ax[1,0].set_ylabel("cumulative evaluation safety violations")
    ax[0,0].legend(fontsize=8); fig.text(.5,.005,"Shaded bands: 95% confidence intervals across five seeds",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.025,1,1)); save_figure(fig,"figure1_learning_comparison")

def figure2(runs):
    fig,ax=plt.subplots(1,3,figsize=(15,5))
    common=json.loads(MANIFEST.read_text())['common']
    goal=(common.get('goal_x',2.),common.get('goal_y',2.))
    for j,c in enumerate(CONDITIONS):
      cp_result=next(d for cc,mm,s,d in runs if cc==c and mm=="cp" and s==representative_seed())
      model=np.load(cp_result["cbvf"]["path"]); oracle=np.load(oracle_cbvf_path(c))
      hi=int(np.argmin(np.abs(model["th_grid"]-np.pi/4)))
      Bm=model["B_table"][:,:,hi,0]; Bt=oracle["B_table"][:,:,hi,0]
      x,y=model["x_grid"],model["y_grid"]
      ax[j].contour(x,y,Bm.T,levels=[0],colors="navy",linewidths=1.8,linestyles="--")
      ax[j].contour(x,y,Bt.T,levels=[0],colors="darkorange",linewidths=1.8)
      ax[j].contour(x,y,Bm.T,levels=[cp_result["offline_cp_qp_feasibility_audit"]["activation_upper"]],colors="darkgreen",linewidths=1.4,linestyles="-.")
      for m in METHODS:
        # Matched-policy counterfactual: one frozen CP-trained actor and one
        # common seeded start, executed through each filter.  This isolates
        # the trajectory-level effect of the safety filter from training-seed
        # differences; the independent-policy comparison remains Figure 1.
        tr=cp_result["final_evaluation"][m]["representative_trajectory"]["states"]
        xy=np.asarray(tr)[:,:2]; ax[j].plot(xy[:,0],xy[:,1],color=COLORS[m],label=m)
      ax[j].add_patch(plt.Circle((0,0),.7,color="firebrick",alpha=.35)); ax[j].add_patch(plt.Circle(goal,.5,color="gold",alpha=.3))
      for radius in [.7+edge for edge in cp_result["mondrian_calibration"]["clearance_edges"]]: ax[j].add_patch(plt.Circle((0,0),radius,fill=False,ls=":",color="gray",lw=.7))
      ax[j].set(xlim=(-4,4),ylim=(-4,4),aspect="equal",title=c.title(),xlabel="x",ylabel="y"); ax[j].grid(alpha=.2)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],color=COLORS[m],label=LABELS[m]) for m in METHODS]
    handles += [Line2D([0],[0],color="navy",ls="--",label="nominal/uncalibrated B=0"),
                Line2D([0],[0],color="darkorange",label="true-model oracle B=0"),
                Line2D([0],[0],color="darkgreen",ls="-.",label="shared filter activation boundary")]
    ax[0].legend(handles=handles,fontsize=7); fig.tight_layout(); save_figure(fig,"figure2_trajectories_boundaries")

def figure3(runs):
    cps=[(c,s,d) for c,m,s,d in runs if m=="cp"]
    fig,ax=plt.subplots(2,3,figsize=(14,7))
    for j,c in enumerate(CONDITIONS):
      d=next(d for cc,s,d in cps if cc==c and s==representative_seed()); st=d["calibration_stats"]
      samples=st["regional_dimensional_score_samples"]
      ax[0,j].boxplot(samples,showfliers=False); ax[0,j].plot(range(1,len(samples)+1),[r["xi_hat_off"] for r in st["regions"]],"r_",ms=14,label="buffer")
      ax[0,j].set_xticks(range(1,len(samples)+1),[f"R{i}\nn={r['n_scores']}" for i,r in enumerate(st["regions"])])
      ax[0,j].set_title(c.title()); ax[0,j].set_ylabel("dimensional score / buffer"); ax[0,j].legend()
      cov=[heldout_coverage(x) for cc,s,x in cps if cc==c]
      ax[1,j].bar([0,1],[.95,np.mean(cov)],color=["#999999",COLORS["cp"]]); ax[1,j].set_xticks([0,1],["target","empirical"]); ax[1,j].set_ylim(0,1.05)
      ax[1,j].scatter(np.ones(len(cov))+np.linspace(-.08,.08,len(cov)),cov,color="black",s=18,zorder=3,label="per seed")
      ax[1,j].legend(fontsize=8)
      ax[1,j].set_ylabel("simultaneous trajectory coverage")
    fig.tight_layout(); save_figure(fig,"figure3_conformal_calibration")

def figure4(runs):
    """Plot saved audit evidence; do not reconstruct margins with stale constants."""
    fig,ax=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for j,c in enumerate(CONDITIONS):
      rr=sorted((seed,d) for cc,m,seed,d in runs if cc==c and m=="cp")
      positions=np.arange(len(rr))
      offline=[d["offline_cp_qp_feasibility_audit"] for _,d in rr]
      ax[0,j].plot(positions,[a["min_feasibility_margin"] for a in offline],"o-",label="calibrated exhaustive minimum")
      ax[0,j].plot(positions,[a["min_uncalibrated_feasibility_margin"] for a in offline],"s--",label="uncalibrated exhaustive minimum")
      ax[0,j].set_title(c.title())
      ax[0,j].set_ylabel("QP feasibility margin")
      runtime=[primary(d,"cp")["min_active_feasibility_margin"] for _,d in rr]
      ax[1,j].plot(positions,runtime,"o-",color=COLORS["cp"],label="minimum on executed CP evaluation states")
      ax[1,j].set_ylabel("Runtime feasibility margin")
      for a in ax[:,j]:
        a.axhline(0,color="firebrick",lw=1)
        a.set_xticks(positions,[str(seed) for seed,_ in rr],rotation=45,ha="right")
        a.set_xlabel("Reserved final seed")
        a.grid(alpha=.2)
        a.legend(fontsize=7)
    save_figure(fig,"figure4_feasibility_verification")

def table_and_figure5(runs):
    fields=[("return","return"),("goal_rate","goal success"),("unsafe_rate","unsafe episode"),("total_safety_violations","safety violations"),("minimum_safety_margin","minimum safety margin"),("minimum_controller_update_safety_margin","controller-update minimum margin"),("minimum_integration_substep_safety_margin","integration-substep minimum margin"),("total_intersample_safety_violations","inter-sample safety violations"),("intervention_rate","intervention rate"),("infeasible_rate_among_active","QP infeasibility"),("rail_fallback_rate_among_infeasible","fallback rate"),("empirical_simultaneous_coverage","coverage"),("C_x","C_x"),("L_Psi_min","L_Psi minimum"),("L_Psi_max","L_Psi maximum"),("epsilon_inter_min","inter-sample tightening minimum"),("epsilon_inter_max","inter-sample tightening maximum")]
    rows=[]
    for c in CONDITIONS:
      for m in METHODS:
        ds=[d for cc,mm,s,d in runs if cc==c and mm==m]; row={"condition":c,"method":m}
        for key,label in fields:
          vals=([heldout_coverage(d) for d in ds] if key=="empirical_simultaneous_coverage" and m=="cp" else [primary(d,m).get(key, float("nan")) for d in ds if math.isfinite(primary(d,m).get(key, float("nan")))])
          mu,sd=mean_sd(vals) if vals else (float("nan"),float("nan")); row[key+"_mean"]=mu; row[key+"_sd"]=sd
        mins=[d["offline_cp_qp_feasibility_audit"]["min_feasibility_margin"] for d in ds]; row["offline_min_margin"]=min(mins); rows.append(row)
    with (OUT/"final_metrics.csv").open("w",newline="") as f:
      w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT/"final_metrics.json").write_text(json.dumps(rows,indent=2,allow_nan=True))
    paper_fields=["condition","method","return_mean_sd","goal_rate_mean_sd","unsafe_rate_mean_sd","total_safety_violations_mean_sd","minimum_safety_margin_mean_sd","intervention_rate_mean_sd"]
    with (OUT/"main_results_table.csv").open("w",newline="") as f:
      w=csv.DictWriter(f,fieldnames=paper_fields); w.writeheader()
      for r in rows:
        fmt=lambda k: f"{r[k+'_mean']:.6g} +/- {r[k+'_sd']:.6g}"
        w.writerow({"condition":r["condition"],"method":LABELS[r["method"]],
          "return_mean_sd":fmt("return"),"goal_rate_mean_sd":fmt("goal_rate"),
          "unsafe_rate_mean_sd":fmt("unsafe_rate"),"total_safety_violations_mean_sd":fmt("total_safety_violations"),
          "minimum_safety_margin_mean_sd":fmt("minimum_safety_margin"),
          "intervention_rate_mean_sd":"not applicable" if r["method"]=="nominal" else fmt("intervention_rate")})
    with (OUT/"main_results_table.tex").open("w") as f:
      f.write("\\begin{tabular}{llrrrrrr}\\toprule\nCondition & Method & Return & Goal & Unsafe & Violations & Min. margin & Intervention \\\\\n\\midrule\n")
      for r in rows:
        fmt=lambda k: f"{r[k+'_mean']:.3f} $\\pm$ {r[k+'_sd']:.3f}"
        intervention="not applicable" if r["method"]=="nominal" else fmt("intervention_rate")
        f.write(f"{r['condition'].title()} & {LABELS[r['method']]} & {fmt('return')} & {fmt('goal_rate')} & {fmt('unsafe_rate')} & {fmt('total_safety_violations')} & {fmt('minimum_safety_margin')} & {intervention} \\\\\n")
      f.write("\\bottomrule\\end{tabular}\n")
    fig,ax=plt.subplots(2,3,figsize=(14,7)); metrics=["goal_rate","unsafe_rate","intervention_rate","return","empirical_simultaneous_coverage","offline_min_margin"]
    for a,key in zip(ax.flat,metrics):
      x=np.arange(3); width=.25
      for i,m in enumerate(METHODS):
        vals=[next(r for r in rows if r["condition"]==c and r["method"]==m).get(key+"_mean",next(r for r in rows if r["condition"]==c and r["method"]==m).get(key)) for c in CONDITIONS]
        a.bar(x+(i-1)*width,vals,width,label=m,color=COLORS[m])
      a.set_xticks(x,CONDITIONS); a.set_title(key.replace("_"," ")); a.grid(axis="y",alpha=.2)
    ax[0,0].legend(); fig.tight_layout(); save_figure(fig,"figure5_safety_performance_summary")

    cert=[]
    for c in CONDITIONS:
      ds=[d for cc,mm,s,d in runs if cc==c and mm=="cp"]
      region_buffers=np.array([[r["xi_hat_off"] for r in d["mondrian_calibration"]["regions"]] for d in ds])
      finals=[primary(d,"cp") for d in ds]
      cert.append({
        "condition":c,
        "empirical_coverage_mean":float(np.mean([heldout_coverage(d) for d in ds])),
        "target_coverage":.95,
        "regional_buffers_mean":";".join(f"{x:.6g}" for x in region_buffers.mean(0)),
        "regional_buffers_sd":";".join(f"{x:.6g}" for x in region_buffers.std(0,ddof=1)),
        "calibrated_qp_infeasibility_rate":float(np.mean([x["infeasible_rate_among_active"] for x in finals])),
        "fallback_rate":float(np.mean([x["rail_fallback_rate_among_infeasible"] for x in finals])),
        "fully_certified_episode_fraction":float(np.mean([1.0 if x["infeasible_total"]==0 and x["unsafe_rate"]==0 else 0.0 for x in finals])),
        "minimum_offline_feasibility_margin":float(min(d["offline_cp_qp_feasibility_audit"]["min_feasibility_margin"] for d in ds)),
        "C_x":float(np.mean([x.get("C_x", float("nan")) for x in finals])),
        "L_Psi_min":float(np.min([x.get("L_Psi_min", float("nan")) for x in finals])),
        "L_Psi_max":float(np.max([x.get("L_Psi_max", float("nan")) for x in finals])),
        "epsilon_inter_min":float(np.min([x.get("epsilon_inter_min", float("nan")) for x in finals])),
        "epsilon_inter_max":float(np.max([x.get("epsilon_inter_max", float("nan")) for x in finals])),
        "minimum_controller_update_safety_margin":float(np.min([x.get("minimum_controller_update_safety_margin", float("nan")) for x in finals])),
        "minimum_integration_substep_safety_margin":float(np.min([x.get("minimum_integration_substep_safety_margin", float("nan")) for x in finals])),
        "total_intersample_safety_violations":float(np.sum([x.get("total_intersample_safety_violations", 0.0) for x in finals])),
      })
    with (OUT/"certification_feasibility_table.csv").open("w",newline="") as f:
      w=csv.DictWriter(f,fieldnames=list(cert[0])); w.writeheader(); w.writerows(cert)
    with (OUT/"certification_feasibility_table.tex").open("w") as f:
      f.write("\\begin{tabular}{lrrlrrrr}\\toprule\nCondition & Coverage & Target & Regional buffers & QP infeas. & Fallback & Certified episodes & Min. offline margin \\\\\n\\midrule\n")
      for r in cert:
        f.write(f"{r['condition'].title()} & {r['empirical_coverage_mean']:.3f} & {r['target_coverage']:.2f} & {r['regional_buffers_mean']} & {r['calibrated_qp_infeasibility_rate']:.3f} & {r['fallback_rate']:.3f} & {r['fully_certified_episode_fraction']:.3f} & {r['minimum_offline_feasibility_margin']:.3e} \\\\\n")
      f.write("\\bottomrule\\end{tabular}\n")

def main():
    global STATE, OUT, MANIFEST
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--results-dir", default=None)
    args=parser.parse_args()
    MANIFEST=Path(args.manifest).resolve()
    manifest=json.loads(MANIFEST.read_text())
    configured=Path(args.results_dir or manifest.get("results_dir", "results"))
    results_dir=configured if configured.is_absolute() else ROOT/configured
    STATE=results_dir/"run_state.json"
    OUT=results_dir/"aggregate"
    OUT.mkdir(parents=True,exist_ok=True); runs=load(); figure1(runs); figure2(runs); figure3(runs); figure4(runs); table_and_figure5(runs)
    print(OUT)
if __name__=="__main__": main()
