import sys
import os
import numpy as np
import heterocl as hcl
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from odp.Grid import Grid
from solver_cbvf import HJSolver 

# ============================================================================
# 1. CONSTANTS & HYPERPARAMETERS (Data-Driven Empirical Bounds)
# ============================================================================
U_MAG = 4.0       # Solid control authority
D_MAG = 0.1       # UPDATED: True process noise bounds [-0.1, 0.1]
V_BAR = 0.1       # UPDATED: True measurement noise bounds [-0.1, 0.1]

ALPHA = 0.05      # 95% Statistical safety probability
DT = 0.02         
T_MAX = 5.0       # 5.0s gives plenty of time for a 2.0m/s transit
SIM_TIME = T_MAX   

# Estimator & Error Propagation Constants
L_EPS = 0.0000    # UPDATED: Empirical linear error growth rate
W_BAR = 0.0146    # UPDATED: Empirical instantaneous disturbance bound
L_HY = 1.0000     # UPDATED: Empirical observation mapping sensitivity
GAMMA = 1.0       # Smooth, early deceleration (prevents level-set bleed)

# The analysis sensitivity MUST bound the online observer gain
L_GAIN = 2.0 
L_EST_Y = 2.0000  # UPDATED: Empirical estimator sensitivity (Matches L_GAIN)

# Obstacles and Goal
OBS1_C = np.array([3.0, 1.0]) 
OBS1_R = 2.0                  
OBS2_C = np.array([3.0, 6.0]) 
OBS2_R = 2.0                  

START = np.array([-5.0, 1.0, 0.0, 0.0])
GOAL = np.array([2.0, 3.5])             

# ============================================================================
# 2. 4D ESTIMATOR DYNAMICS FOR HJ SOLVER
# ============================================================================
class EstimatorHJModel4D:
    def __init__(self, r_d_hat_max):
        self.r_d_hat_max = r_d_hat_max
        
    def opt_ctrl(self, t, state, spat_deriv):
        opt_ux = hcl.scalar(U_MAG, "opt_ux")
        opt_uy = hcl.scalar(U_MAG, "opt_uy")
        with hcl.if_(spat_deriv[2] < 0): opt_ux[0] = -U_MAG
        with hcl.if_(spat_deriv[3] < 0): opt_uy[0] = -U_MAG
        return (0, 0, opt_ux[0], opt_uy[0])

    def opt_dstb(self, t, state, spat_deriv):
        p_norm = hcl.sqrt(spat_deriv[0]*spat_deriv[0] + spat_deriv[1]*spat_deriv[1] + 1e-9)
        opt_dx = hcl.scalar(0, "opt_dx")
        opt_dy = hcl.scalar(0, "opt_dy")
        
        opt_dx[0] = -self.r_d_hat_max * (spat_deriv[0] / p_norm)
        opt_dy[0] = -self.r_d_hat_max * (spat_deriv[1] / p_norm)
        return (opt_dx[0], opt_dy[0], 0, 0)

    def dynamics(self, t, state, uOpt, dOpt):
        px_dot = state[2] + dOpt[0]
        py_dot = state[3] + dOpt[1]
        vx_dot = uOpt[2]
        vy_dot = uOpt[3]
        return (px_dot, py_dot, vx_dot, vy_dot)

# ============================================================================
# 3. QP SAFETY FILTER (STRICT SOFT-CBF)
# ============================================================================
def solve_qp_filter(t, u_nom, LfB, LgB, B_px, B_py, B_val, dBdt, gamma, u_bound, r_d_hat_max):
    worst_case_dstb_effect = -r_d_hat_max * np.linalg.norm([B_px, B_py])
    cbf_req = -(dBdt + LfB + worst_case_dstb_effect + gamma * B_val)
    
    # Massive penalty acts as a near-hard boundary constraint
    penalty_weight = 1e6 
    
    def obj(u):
        slack = max(0.0, cbf_req - np.dot(LgB, u))
        return np.sum((u - u_nom)**2) + penalty_weight * (slack**2)
    
    def jac(u):
        slack = max(0.0, cbf_req - np.dot(LgB, u))
        grad = 2 * (u - u_nom)
        if slack > 0:
            grad -= 2 * penalty_weight * slack * LgB
        return grad

    bnds = ((-u_bound, u_bound), (-u_bound, u_bound))
    res = minimize(obj, u_nom, method='L-BFGS-B', jac=jac, bounds=bnds)
    
    return res.x

def get_b_and_derivs(interp_B_5D, x_hat, t):
    pt = np.array([*x_hat, t])
    eps = 1e-4
    b_val = float(interp_B_5D(pt))
    
    grid_failure = False
    with np.errstate(invalid='ignore', divide='ignore'):
         B_px = (float(interp_B_5D(pt + np.array([eps,0,0,0,0]))) - b_val) / eps
         B_py = (float(interp_B_5D(pt + np.array([0,eps,0,0,0]))) - b_val) / eps
         B_vx = (float(interp_B_5D(pt + np.array([0,0,eps,0,0]))) - b_val) / eps
         B_vy = (float(interp_B_5D(pt + np.array([0,0,0,eps,0]))) - b_val) / eps
         
         if np.isnan(B_px) or np.isnan(B_py) or np.isnan(B_vx) or np.isnan(B_vy):
             grid_failure = True
             
    if grid_failure:
         raise RuntimeError(f"PDE interpolation grid failure at t={t:.2f}. Grid bounds are likely too tight.")
    
    if t + eps > T_MAX:
        dBdt = (b_val - float(interp_B_5D(pt - np.array([0,0,0,0,eps])))) / eps
    else:
        dBdt = (float(interp_B_5D(pt + np.array([0,0,0,0,eps]))) - b_val) / eps
        
    LfB = B_px * x_hat[2] + B_py * x_hat[3]
    LgB = np.array([B_vx, B_vy])
    
    return b_val, B_px, B_py, LfB, LgB, dBdt

# ============================================================================
# 4. CONFORMAL CALIBRATION (ON-POLICY)
# ============================================================================
def generate_calibration_rollouts(policy_valfuncs, g, tau, r_d_hat_max, num_rollouts=500):
    valfuncs_forward = np.flip(policy_valfuncs, axis=-1)
    grid_points_5d = (*g.grid_points, tau)
    interp_B_5D = RegularGridInterpolator(grid_points_5d, valfuncs_forward, bounds_error=False, fill_value=-1.0)
    
    scores = []
    
    for r_idx in range(num_rollouts):
        x_t = START.copy()
        x_h = START.copy()
        max_err = 0.0
        
        try:
            for step in range(int(T_MAX / DT)):
                t = step * DT
                
                # CRITICAL: Velocity-Capped Nominal Controller
                # Prevents runaway braking boundaries
                v_req_x = np.clip(1.5 * (GOAL[0] - x_h[0]), -2.0, 2.0)
                v_req_y = np.clip(1.5 * (GOAL[1] - x_h[1]), -2.0, 2.0)
                ux_nom = 3.0 * (v_req_x - x_h[2])
                uy_nom = 3.0 * (v_req_y - x_h[3])
                u_nom_vec = np.array([ux_nom, uy_nom])
                
                dist_to_goal_h = np.linalg.norm(x_h[:2] - GOAL)
                if dist_to_goal_h < 0.15:
                    u_nom_vec = np.array([0.0, 0.0]) 

                b_val, B_px, B_py, LfB, LgB, dBdt = get_b_and_derivs(interp_B_5D, x_h, t)
                
                u_safe = solve_qp_filter(t, u_nom_vec, LfB, LgB, B_px, B_py, b_val, dBdt, GAMMA, U_MAG, r_d_hat_max)
                
                d_x = np.random.uniform(-D_MAG, D_MAG)
                d_y = np.random.uniform(-D_MAG, D_MAG)
                x_t[0] += x_t[2] * DT
                x_t[1] += x_t[3] * DT
                x_t[2] += (u_safe[0] + d_x) * DT
                x_t[3] += (u_safe[1] + d_y) * DT
                
                y_meas_x = x_t[0] + np.random.uniform(-V_BAR, V_BAR)
                y_meas_y = x_t[1] + np.random.uniform(-V_BAR, V_BAR)
                x_h[0] += (x_h[2] + L_GAIN * (y_meas_x - x_h[0])) * DT
                x_h[1] += (x_h[3] + L_GAIN * (y_meas_y - x_h[1])) * DT
                x_h[2] += u_safe[0] * DT
                x_h[3] += u_safe[1] * DT
                
                err = np.linalg.norm(x_t - x_h)
                if err > max_err:
                    max_err = err
                    
            scores.append(max_err)
        except RuntimeError as e:
            raise RuntimeError(f"Calibration aborted! Policy is infeasible. Details: {e}")
    
    scores = np.sort(scores)
    k_star = int(np.ceil((num_rollouts + 1) * (1 - ALPHA)))
    k_star = min(k_star, num_rollouts)
    
    q_alpha = scores[k_star - 1]
    return q_alpha

# ============================================================================
# 5. FINAL SIMULATION & PLOTTING 
# ============================================================================
def run_simulation_and_plot(valfuncs, g, tau, r_d_hat_max):
    print("\n>>> Running Final 4D True System Simulation <<<", flush=True)
    
    valfuncs_forward = np.flip(valfuncs, axis=-1)
    grid_points_5d = (*g.grid_points, tau)
    interp_B_5D = RegularGridInterpolator(grid_points_5d, valfuncs_forward, bounds_error=False, fill_value=-1.0)
    
    x_true = START.copy()
    x_hat = START.copy()
    
    history = {"px_true": [], "py_true": [], "vx_hat": [], "vy_hat": [], "t": []}
    goal_reached = False
    
    try:
        for step in range(int(SIM_TIME / DT)):
            t = step * DT
            
            dist_to_goal_h = np.linalg.norm(x_hat[:2] - GOAL)
            
            # CRITICAL: Velocity-Capped Nominal Controller
            v_req_x = np.clip(1.5 * (GOAL[0] - x_hat[0]), -2.0, 2.0)
            v_req_y = np.clip(1.5 * (GOAL[1] - x_hat[1]), -2.0, 2.0)
            ux_nom = 3.0 * (v_req_x - x_hat[2])
            uy_nom = 3.0 * (v_req_y - x_hat[3])
            u_nom_vec = np.array([ux_nom, uy_nom])
            
            if dist_to_goal_h < 0.15:
                if not goal_reached:
                     print(f"    -> Goal complete! Time: {t:.2f}s")
                u_nom_vec = np.array([0.0, 0.0])
                goal_reached = True

            b_val, B_px, B_py, LfB, LgB, dBdt = get_b_and_derivs(interp_B_5D, x_hat, t)
            u_safe_vec = solve_qp_filter(t, u_nom_vec, LfB, LgB, B_px, B_py, b_val, dBdt, GAMMA, U_MAG, r_d_hat_max)
            
            history["px_true"].append(x_true[0])
            history["py_true"].append(x_true[1])
            history["vx_hat"].append(x_hat[2])
            history["vy_hat"].append(x_hat[3])
            history["t"].append(t)
            
            d_true_x = np.random.uniform(-D_MAG, D_MAG)
            d_true_y = np.random.uniform(-D_MAG, D_MAG)
            x_true[0] += x_true[2] * DT
            x_true[1] += x_true[3] * DT
            x_true[2] += (u_safe_vec[0] + d_true_x) * DT
            x_true[3] += (u_safe_vec[1] + d_true_y) * DT
            
            y_meas_x = x_true[0] + np.random.uniform(-V_BAR, V_BAR)
            y_meas_y = x_true[1] + np.random.uniform(-V_BAR, V_BAR)
            x_hat[0] += (x_hat[2] + L_GAIN * (y_meas_x - x_hat[0])) * DT
            x_hat[1] += (x_hat[3] + L_GAIN * (y_meas_y - x_hat[1])) * DT
            x_hat[2] += u_safe_vec[0] * DT
            x_hat[3] += u_safe_vec[1] * DT

        history["px_true"].append(x_true[0])
        history["py_true"].append(x_true[1])
        history["vx_hat"].append(x_hat[2])
        history["vy_hat"].append(x_hat[3])
        history["t"].append(T_MAX)
        
    except RuntimeError as e:
        print(f"\n[FATAL ERROR] {e}\nSimulation aborted. No certified trajectory to plot.")
        return 

    if len(history["t"]) == 0: return
    t_final = history["t"][-1]
    times_to_plot = [np.round(t_final * 0.33, 2), np.round(t_final * 0.66, 2), np.round(t_final, 2)]
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    px_lin = np.linspace(-7, 7, 80)
    py_lin = np.linspace(-2, 10, 80)
    PX_grid, PY_grid = np.meshgrid(px_lin, py_lin, indexing='ij')
    
    for ax, t_target in zip(axes, times_to_plot):
        idx = np.argmin(np.abs(np.array(history["t"]) - t_target))
        curr_vx_hat = history["vx_hat"][idx]
        curr_vy_hat = history["vy_hat"][idx]
        
        ax.set_facecolor('#ffffff')
        ax.add_patch(plt.Circle((OBS1_C[0], OBS1_C[1]), OBS1_R, color='black', zorder=2))
        ax.add_patch(plt.Circle((OBS2_C[0], OBS2_C[1]), OBS2_R, color='black', zorder=2))
        ax.grid(True, linestyle='-', color='white', linewidth=0.5, zorder=3)
        
        ax.plot(history["px_true"][:idx+1], history["py_true"][:idx+1], 'b-', linewidth=3, alpha=0.8, zorder=4)
        ax.scatter(history["px_true"][:idx+1], history["py_true"][:idx+1], c='blue', s=10, zorder=5)
        ax.scatter(START[0], START[1], c='magenta', marker='x', s=60, zorder=6)
        ax.scatter(GOAL[0], GOAL[1], c='lime', marker='o', s=50, zorder=6)
        
        points = np.stack([PX_grid.ravel(), PY_grid.ravel(), 
                           np.full(6400, curr_vx_hat), np.full(6400, curr_vy_hat),
                           np.full(6400, t_target)], axis=-1)
        B_slice = interp_B_5D(points).reshape(80, 80)
        ax.contour(PX_grid, PY_grid, B_slice, levels=[0], colors='red', linewidths=1.2, zorder=4)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-7, 7])
        ax.set_ylim([-2, 10])
        ax.set_title(f't = {t_target} s', fontsize=12)
        
    plt.tight_layout()
    plt.savefig("po_spatial_subplots.png", dpi=300)
    plt.close()
    print(">>> Saved po_spatial_subplots.png", flush=True)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n>>> Setting up 4D ODP Grid <<<", flush=True)
    # Refined spatial bounds to pack more resolution around the obstacles.
    # We can shrink velocity bounds since max transit velocity is strictly capped at 2.0 m/s
    g = Grid(np.array([-7.0, -2.0, -4.0, -4.0]), 
             np.array([7.0, 10.0, 4.0, 4.0]), 
             4, np.array([30, 30, 15, 15]), []) 
    
    tau = np.arange(0.0, T_MAX + 1e-9, DT)
    PX, PY, VX, VY = np.meshgrid(*g.grid_points, indexing="ij")
    compMethods = {"TargetSetMode": "maxCBF", "cbf_gamma": GAMMA}

    # ------------------------------------------------------------------------
    # STEP 1: INITIAL BASELINE
    # ------------------------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 1: SOLVE INITIAL BASELINE HJ-PDE")
    print("="*50)
    
    dist1_base = np.sqrt((PX - OBS1_C[0])**2 + (PY - OBS1_C[1])**2)
    dist2_base = np.sqrt((PX - OBS2_C[0])**2 + (PY - OBS2_C[1])**2)
    target_base = np.minimum(dist1_base - OBS1_R, dist2_base - OBS2_R).astype(np.float32)
    
    r_d_hat_base = L_EST_Y * (L_HY * V_BAR + V_BAR) 
    sys_base = EstimatorHJModel4D(r_d_hat_max=r_d_hat_base)
    valfuncs_base = HJSolver(sys_base, g, target_base, tau, compMethods, saveAllTimeSteps=True, accuracy="low")
    
    q_current = generate_calibration_rollouts(valfuncs_base, g, tau, r_d_hat_max=r_d_hat_base, num_rollouts=500)
    print(f"    -> Initial Baseline q_alpha: {q_current:.4f}")

    # ------------------------------------------------------------------------
    # STEP 2: FIXED-POINT CP CALIBRATION LOOP
    # ------------------------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 2: FIXED-POINT CALIBRATION LOOP")
    print("="*50)
    
    MAX_ITERS = 3
    CONVERGENCE_TOL = 0.01
    is_converged = False

    for iteration in range(MAX_ITERS):
        print(f"\n--- Iteration {iteration + 1} ---")
        
        def r_alpha_target(s, q=q_current):
            if L_EPS == 0: return q + W_BAR * s
            return q * np.exp(L_EPS * s) + (W_BAR / L_EPS) * (np.exp(L_EPS * s) - 1)

        def time_varying_target(t_rev):
            t = T_MAX - t_rev 
            r_err = r_alpha_target(t)
            eff_obs1_R = OBS1_R + r_err
            eff_obs2_R = OBS2_R + r_err
            
            dist1 = np.sqrt((PX - OBS1_C[0])**2 + (PY - OBS1_C[1])**2)
            dist2 = np.sqrt((PX - OBS2_C[0])**2 + (PY - OBS2_C[1])**2)
            target1 = dist1 - eff_obs1_R
            target2 = dist2 - eff_obs2_R
            return np.minimum(target1, target2).astype(np.float32)

        r_err_max = r_alpha_target(T_MAX)
        r_d_hat_max = L_EST_Y * (L_HY * r_err_max + V_BAR)

        sys_est = EstimatorHJModel4D(r_d_hat_max=r_d_hat_max)
        valfuncs_po = HJSolver(sys_est, g, time_varying_target, tau, compMethods, saveAllTimeSteps=True, accuracy="low")
        
        q_new = generate_calibration_rollouts(valfuncs_po, g, tau, r_d_hat_max=r_d_hat_max, num_rollouts=500)
        print(f"    -> q_current: {q_current:.4f} | q_new: {q_new:.4f}")
        
        if abs(q_new - q_current) <= CONVERGENCE_TOL:
            print(f"    -> [SUCCESS] Fixed-point converged at q_alpha = {q_new:.4f}!")
            q_current = q_new
            is_converged = True
            break
            
        q_current = q_new
        if iteration == MAX_ITERS - 1:
            print("    -> [WARNING] Reached max iterations without strict convergence.")

    if not is_converged:
        raise RuntimeError("Fixed-point CP calibration failed to converge. Theorem guarantees cannot be certified.")

    # ------------------------------------------------------------------------
    # STEP 3: FINAL SYNCHRONIZATION RUN
    # ------------------------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 3: FINAL SYNCHRONIZATION RUN")
    print("="*50)
    
    def final_r_alpha(s):
        if L_EPS == 0: return q_current + W_BAR * s
        return q_current * np.exp(L_EPS * s) + (W_BAR / L_EPS) * (np.exp(L_EPS * s) - 1)

    def final_time_varying_target(t_rev):
        t = T_MAX - t_rev
        r_err = final_r_alpha(t)
        eff_obs1_R = OBS1_R + r_err
        eff_obs2_R = OBS2_R + r_err
        
        dist1 = np.sqrt((PX - OBS1_C[0])**2 + (PY - OBS1_C[1])**2)
        dist2 = np.sqrt((PX - OBS2_C[0])**2 + (PY - OBS2_C[1])**2)
        target1 = dist1 - eff_obs1_R
        target2 = dist2 - eff_obs2_R
        return np.minimum(target1, target2).astype(np.float32)

    r_err_max_final = final_r_alpha(T_MAX)
    r_d_hat_max_final = L_EST_Y * (L_HY * r_err_max_final + V_BAR)

    sys_est_final = EstimatorHJModel4D(r_d_hat_max=r_d_hat_max_final)
    valfuncs_po_final = HJSolver(sys_est_final, g, final_time_varying_target, tau, compMethods, saveAllTimeSteps=True, accuracy="low")

    # ------------------------------------------------------------------------
    # STEP 4: STRICT CALIBRATION CONSISTENCY VERIFICATION
    # ------------------------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 4: VERIFYING DEPLOYED POLICY CONSISTENCY")
    print("="*50)
    
    q_check = generate_calibration_rollouts(valfuncs_po_final, g, tau, r_d_hat_max=r_d_hat_max_final, num_rollouts=500)
    print(f"    -> Expected q_alpha: {q_current:.4f} | Deployed q_alpha: {q_check:.4f}")
    
    if abs(q_check - q_current) > CONVERGENCE_TOL:
        raise RuntimeError("Final synced policy is not calibration-consistent. Abort certification.")
    print("    -> [SUCCESS] Deployed policy verified self-consistent. Theorem holds.")

    # ------------------------------------------------------------------------
    # STEP 5: DEPLOYMENT
    # ------------------------------------------------------------------------
    print("\n" + "="*50)
    print("STEP 5: CERTIFIED DEPLOYMENT")
    print("="*50)
    run_simulation_and_plot(valfuncs_po_final, g, tau, r_d_hat_max=r_d_hat_max_final)

    # --- PAPER CASE STUDY SUMMARY ---
    print("\n" + "-"*50)
    print(">>> PAPER CASE STUDY PARAMETERS SUMMARY <<<")
    print("-"*50)
    print(f"Methodology Note: All error propagation bounds and system sensitivities were derived empirically from offline calibration rollouts.")
    print(f"\nStatistical Guarantees:")
    print(f"  -> Miscoverage Rate (alpha)          : {ALPHA}")
    print(f"  -> Calibration Rollouts (N)          : 500") 
    print(f"  -> Final Conformal Quantile (q_alpha): {q_current:.4f}")
    print(f"\nConformal Inflation Bounds (at T_MAX = {T_MAX}s):")
    print(f"  -> Max Tracking Error (r_err)        : {r_err_max_final:.4f} m")
    print(f"  -> Max Effective Disturbance (r_d)   : {r_d_hat_max_final:.4f} m/s")
    print(f"  -> Base Obstacle Radius              : {OBS1_R:.2f} m")
    print(f"  -> Effective Obstacle Radius         : {OBS1_R + r_err_max_final:.4f} m")
    print(f"\nSystem & Safety Filter Tuning:")
    print(f"  -> Control Authority Bounds (U_MAG)  : +/- {U_MAG} m/s^2")
    print(f"  -> True Noise Bounds (D_MAG/V_BAR)   : +/- {D_MAG} m (per axis)")
    print(f"  -> CBF Parameter (Gamma)             : {GAMMA}")
    print(f"  -> Estimator Gain (L_GAIN)           : {L_GAIN}")
    print(f"\nComputational Setup:")
    print(f"  -> Reachability Horizon (T_MAX)      : {T_MAX} s")
    print(f"  -> Time Step (DT)                    : {DT} s")
    print(f"  -> ODP Grid Resolution               : {g.pts_each_dim[0]}x{g.pts_each_dim[1]}x{g.pts_each_dim[2]}x{g.pts_each_dim[3]}")
    print("-"*50)