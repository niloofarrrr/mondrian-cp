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

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
np.random.seed(7)

# -----------------------------------------------------------------------------
# DESIGN / CERTIFICATION SIDE
# -----------------------------------------------------------------------------
U_MAG = 4.0
ALPHA = 0.05
DT = 0.02
T_MAX = 5.0
SIM_TIME = T_MAX
GAMMA = 1.0

D_MAG_DESIGN = 0.1
V_BAR_DESIGN = 0.1
L_GAIN_DESIGN = 2.0

# Error propagation constants used in the design
L_EPS = 0.0000
W_BAR = 0.0146
L_HY = 1.0000
L_EST_Y = 2.0000

# -----------------------------------------------------------------------------
# DEPLOYMENT SIDE FOR FINAL MONTE-CARLO FIGURE
# Intentionally harsher than design to produce some failures
# -----------------------------------------------------------------------------
DEPLOY_D_MAG = 0.24
DEPLOY_V_BAR = 0.24
DEPLOY_L_GAIN = 0.45

DROPOUT_PROB = 0.06
DROPOUT_STEPS = 18
OUTLIER_PROB = 0.05
OUTLIER_SCALE = 6.0

NUM_PLOT_ROLLOUTS = 10

DISTURBANCE_BURST_PROB = 0.05
DISTURBANCE_BURST = 0.22

INIT_EST_POS_BIAS = 0.45
INIT_EST_VEL_BIAS = 0.30

# -----------------------------------------------------------------------------
# GEOMETRY
# -----------------------------------------------------------------------------
OBS1_C = np.array([3.0, 1.0])
OBS1_R = 2.0
OBS2_C = np.array([3.0, 6.0])
OBS2_R = 2.0

START = np.array([-5.0, 1.0, 0.0, 0.0])
GOAL = np.array([2.0, 3.5])

# =============================================================================
# HJ ESTIMATOR-SPACE MODEL
# =============================================================================
class EstimatorHJModel4D:
    def __init__(self, r_d_hat_max):
        self.r_d_hat_max = r_d_hat_max

    def opt_ctrl(self, t, state, spat_deriv):
        opt_ux = hcl.scalar(U_MAG, "opt_ux")
        opt_uy = hcl.scalar(U_MAG, "opt_uy")

        with hcl.if_(spat_deriv[2] < 0):
            opt_ux[0] = -U_MAG
        with hcl.if_(spat_deriv[3] < 0):
            opt_uy[0] = -U_MAG

        return (0, 0, opt_ux[0], opt_uy[0])

    def opt_dstb(self, t, state, spat_deriv):
        p_norm = hcl.sqrt(
            spat_deriv[0] * spat_deriv[0]
            + spat_deriv[1] * spat_deriv[1]
            + 1e-9
        )

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

# =============================================================================
# HELPERS
# =============================================================================
def physical_collision(pos):
    d1 = np.linalg.norm(pos - OBS1_C)
    d2 = np.linalg.norm(pos - OBS2_C)
    return (d1 <= OBS1_R) or (d2 <= OBS2_R)

def nominal_controller(x_hat):
    v_req_x = np.clip(1.5 * (GOAL[0] - x_hat[0]), -2.0, 2.0)
    v_req_y = np.clip(1.5 * (GOAL[1] - x_hat[1]), -2.0, 2.0)

    ux_nom = 3.0 * (v_req_x - x_hat[2])
    uy_nom = 3.0 * (v_req_y - x_hat[3])
    u_nom = np.array([ux_nom, uy_nom])

    if np.linalg.norm(x_hat[:2] - GOAL) < 0.15:
        u_nom = np.array([0.0, 0.0])

    return u_nom

def solve_qp_filter(u_nom, LfB, LgB, B_px, B_py, B_val, dBdt,
                    gamma, u_bound, r_d_hat_max):
    worst_case_dstb_effect = -r_d_hat_max * np.linalg.norm([B_px, B_py])
    cbf_req = -(dBdt + LfB + worst_case_dstb_effect + gamma * B_val)

    penalty_weight = 1e6

    def obj(u):
        slack = max(0.0, cbf_req - np.dot(LgB, u))
        return np.sum((u - u_nom) ** 2) + penalty_weight * (slack ** 2)

    def jac(u):
        slack = max(0.0, cbf_req - np.dot(LgB, u))
        grad = 2.0 * (u - u_nom)
        if slack > 0:
            grad -= 2.0 * penalty_weight * slack * LgB
        return grad

    bounds = ((-u_bound, u_bound), (-u_bound, u_bound))
    res = minimize(obj, u_nom, method="L-BFGS-B", jac=jac, bounds=bounds)
    return res.x

def get_b_and_derivs(interp_B_5D, x_hat, t):
    pt = np.array([x_hat[0], x_hat[1], x_hat[2], x_hat[3], t])
    eps = 1e-4

    b_val = float(interp_B_5D(pt))

    with np.errstate(invalid="ignore", divide="ignore"):
        B_px = (float(interp_B_5D(pt + np.array([eps, 0, 0, 0, 0]))) - b_val) / eps
        B_py = (float(interp_B_5D(pt + np.array([0, eps, 0, 0, 0]))) - b_val) / eps
        B_vx = (float(interp_B_5D(pt + np.array([0, 0, eps, 0, 0]))) - b_val) / eps
        B_vy = (float(interp_B_5D(pt + np.array([0, 0, 0, eps, 0]))) - b_val) / eps

    if np.isnan(B_px) or np.isnan(B_py) or np.isnan(B_vx) or np.isnan(B_vy):
        raise RuntimeError(f"Interpolation failure at t={t:.3f}")

    if t + eps > T_MAX:
        dBdt = (b_val - float(interp_B_5D(pt - np.array([0, 0, 0, 0, eps])))) / eps
    else:
        dBdt = (float(interp_B_5D(pt + np.array([0, 0, 0, 0, eps]))) - b_val) / eps

    LfB = B_px * x_hat[2] + B_py * x_hat[3]
    LgB = np.array([B_vx, B_vy])

    return b_val, B_px, B_py, LfB, LgB, dBdt

def make_interpolator(valfuncs, g, tau):
    valfuncs_forward = np.flip(valfuncs, axis=-1)
    grid_points_5d = (*g.grid_points, tau)
    return RegularGridInterpolator(
        grid_points_5d,
        valfuncs_forward,
        bounds_error=False,
        fill_value=-1.0
    )

def cp_radius_at_time(t, q_alpha_final):
    if L_EPS == 0:
        return q_alpha_final + W_BAR * t
    return (
        q_alpha_final * np.exp(L_EPS * t)
        + (W_BAR / L_EPS) * (np.exp(L_EPS * t) - 1.0)
    )

# =============================================================================
# CALIBRATION ROLLOUTS ON DESIGN SIDE
# =============================================================================
def generate_calibration_rollouts(policy_valfuncs, g, tau, r_d_hat_max, num_rollouts=500):
    interp_B_5D = make_interpolator(policy_valfuncs, g, tau)
    scores = []

    for _ in range(num_rollouts):
        x_t = START.copy()
        x_h = START.copy()
        max_err = 0.0

        for step in range(int(T_MAX / DT)):
            t = step * DT

            u_nom = nominal_controller(x_h)

            b_val, B_px, B_py, LfB, LgB, dBdt = get_b_and_derivs(interp_B_5D, x_h, t)
            u_safe = solve_qp_filter(
                u_nom=u_nom,
                LfB=LfB,
                LgB=LgB,
                B_px=B_px,
                B_py=B_py,
                B_val=b_val,
                dBdt=dBdt,
                gamma=GAMMA,
                u_bound=U_MAG,
                r_d_hat_max=r_d_hat_max
            )

            d_x = np.random.uniform(-D_MAG_DESIGN, D_MAG_DESIGN)
            d_y = np.random.uniform(-D_MAG_DESIGN, D_MAG_DESIGN)

            x_t[0] += x_t[2] * DT
            x_t[1] += x_t[3] * DT
            x_t[2] += (u_safe[0] + d_x) * DT
            x_t[3] += (u_safe[1] + d_y) * DT

            y_meas_x = x_t[0] + np.random.uniform(-V_BAR_DESIGN, V_BAR_DESIGN)
            y_meas_y = x_t[1] + np.random.uniform(-V_BAR_DESIGN, V_BAR_DESIGN)

            x_h[0] += (x_h[2] + L_GAIN_DESIGN * (y_meas_x - x_h[0])) * DT
            x_h[1] += (x_h[3] + L_GAIN_DESIGN * (y_meas_y - x_h[1])) * DT
            x_h[2] += u_safe[0] * DT
            x_h[3] += u_safe[1] * DT

            err = np.linalg.norm(x_t - x_h)
            if err > max_err:
                max_err = err

        scores.append(max_err)

    scores = np.sort(scores)
    k_star = int(np.ceil((num_rollouts + 1) * (1 - ALPHA)))
    k_star = min(k_star, num_rollouts)
    q_alpha = scores[k_star - 1]
    return q_alpha

# =============================================================================
# SINGLE DEPLOYMENT ROLLOUT
# =============================================================================
def simulate_one_rollout(valfuncs, g, tau, r_d_hat_max, q_alpha_final,
                         d_mag, v_bar, l_gain, seed=None):
    if seed is not None:
        np.random.seed(seed)

    interp_B_5D = make_interpolator(valfuncs, g, tau)

    x_true = START.copy()
    x_hat = START.copy()

    # Small initial estimator mismatch
    x_hat[:2] += np.random.uniform(-INIT_EST_POS_BIAS, INIT_EST_POS_BIAS, size=2)
    x_hat[2:] += np.random.uniform(-INIT_EST_VEL_BIAS, INIT_EST_VEL_BIAS, size=2)

    hist = {
        "t": [],
        "px_true": [],
        "py_true": [],
        "px_hat": [],
        "py_hat": [],
        "hit_obstacle": False,
        "reached_goal": False,
        "cp_violated": False
    }

    dropout_remaining = 0

    for step in range(int(SIM_TIME / DT)):
        t = step * DT

        hist["t"].append(t)
        hist["px_true"].append(x_true[0])
        hist["py_true"].append(x_true[1])
        hist["px_hat"].append(x_hat[0])
        hist["py_hat"].append(x_hat[1])

        if physical_collision(x_true[:2]):
            hist["hit_obstacle"] = True
            break

        if np.linalg.norm(x_true[:2] - GOAL) < 0.35:
            hist["reached_goal"] = True
            break

        u_nom = nominal_controller(x_hat)

        b_val, B_px, B_py, LfB, LgB, dBdt = get_b_and_derivs(interp_B_5D, x_hat, t)
        u_safe = solve_qp_filter(
            u_nom=u_nom,
            LfB=LfB,
            LgB=LgB,
            B_px=B_px,
            B_py=B_py,
            B_val=b_val,
            dBdt=dBdt,
            gamma=GAMMA,
            u_bound=U_MAG,
            r_d_hat_max=r_d_hat_max
        )

        # True plant disturbance, harsher than design
        d_true_x = np.random.uniform(-d_mag, d_mag)
        d_true_y = np.random.uniform(-d_mag, d_mag)

        if np.random.rand() < DISTURBANCE_BURST_PROB:
            d_true_x += np.random.choice([-1.0, 1.0]) * DISTURBANCE_BURST
            d_true_y += np.random.choice([-1.0, 1.0]) * DISTURBANCE_BURST

        x_true[0] += x_true[2] * DT
        x_true[1] += x_true[3] * DT
        x_true[2] += (u_safe[0] + d_true_x) * DT
        x_true[3] += (u_safe[1] + d_true_y) * DT

        # Measurement model with occasional dropout / outlier
        if dropout_remaining > 0:
            y_meas_x = x_hat[0]
            y_meas_y = x_hat[1]
            dropout_remaining -= 1
        else:
            if np.random.rand() < DROPOUT_PROB:
                dropout_remaining = DROPOUT_STEPS - 1
                y_meas_x = x_hat[0]
                y_meas_y = x_hat[1]
            else:
                y_meas_x = x_true[0] + np.random.uniform(-v_bar, v_bar)
                y_meas_y = x_true[1] + np.random.uniform(-v_bar, v_bar)

                if np.random.rand() < OUTLIER_PROB:
                    y_meas_x += np.random.choice([-1.0, 1.0]) * OUTLIER_SCALE * v_bar
                    y_meas_y += np.random.choice([-1.0, 1.0]) * OUTLIER_SCALE * v_bar

        # Online estimator is weaker than design
        x_hat[0] += (x_hat[2] + l_gain * (y_meas_x - x_hat[0])) * DT
        x_hat[1] += (x_hat[3] + l_gain * (y_meas_y - x_hat[1])) * DT
        x_hat[2] += u_safe[0] * DT
        x_hat[3] += u_safe[1] * DT

        # Check violation of propagated CP tube
        r_now = cp_radius_at_time(t, q_alpha_final)
        est_err = np.linalg.norm(x_true - x_hat)
        if est_err > r_now:
            hist["cp_violated"] = True

    hist["t"].append(min(SIM_TIME, len(hist["t"]) * DT))
    hist["px_true"].append(x_true[0])
    hist["py_true"].append(x_true[1])
    hist["px_hat"].append(x_hat[0])
    hist["py_hat"].append(x_hat[1])

    if physical_collision(x_true[:2]):
        hist["hit_obstacle"] = True
    if np.linalg.norm(x_true[:2] - GOAL) < 0.35:
        hist["reached_goal"] = True

    # Failure means only obstacle hit
    hist["success"] = (not hist["hit_obstacle"])
    return hist

# =============================================================================
# PLOTTING: THREE PANELS
# =============================================================================
def plot_ten_rollouts_three_panels(valfuncs, g, tau, r_d_hat_max, q_alpha_final):
    # simulate the 10 rollouts once, then reuse them in all 3 panels
    all_histories = []
    success_count = 0
    fail_count = 0
    cp_viol_count = 0

    for i in range(NUM_PLOT_ROLLOUTS):
        hist = simulate_one_rollout(
            valfuncs=valfuncs,
            g=g,
            tau=tau,
            r_d_hat_max=r_d_hat_max,
            q_alpha_final=q_alpha_final,
            d_mag=DEPLOY_D_MAG,
            v_bar=DEPLOY_V_BAR,
            l_gain=DEPLOY_L_GAIN,
            seed=100 + i
        )
        all_histories.append(hist)

        if hist["cp_violated"]:
            cp_viol_count += 1
        if hist["success"]:
            success_count += 1
        else:
            fail_count += 1

    times_to_plot = [1.65, 3.30, 5.00]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    interp_B_5D = make_interpolator(valfuncs, g, tau)
    px_lin = np.linspace(-7, 7, 120)
    py_lin = np.linspace(-2, 10, 120)
    PXs, PYs = np.meshgrid(px_lin, py_lin, indexing="ij")

    for ax, t_target in zip(axes, times_to_plot):
        ax.set_aspect("equal", adjustable="box")

        # physical obstacles
        ax.add_patch(plt.Circle((OBS1_C[0], OBS1_C[1]), OBS1_R,
                                color="black", alpha=0.90, zorder=2))
        ax.add_patch(plt.Circle((OBS2_C[0], OBS2_C[1]), OBS2_R,
                                color="black", alpha=0.90, zorder=2))

        # zero level set slice at this time
        vx_slice = 0.0
        vy_slice = 0.0
        pts = np.stack([
            PXs.ravel(),
            PYs.ravel(),
            np.full(PXs.size, vx_slice),
            np.full(PXs.size, vy_slice),
            np.full(PXs.size, t_target)
        ], axis=-1)

        B_slice = interp_B_5D(pts).reshape(PXs.shape)
        ax.contour(PXs, PYs, B_slice, levels=[0], colors="red", linewidths=2.0, zorder=3)

        # start and goal
        ax.scatter(START[0], START[1], c="magenta", marker="x", s=110, zorder=7)
        ax.scatter(GOAL[0], GOAL[1], c="lime", marker="o", s=80, zorder=7)

        # trajectories up to t_target
        for hist in all_histories:
            t_arr = np.array(hist["t"])
            px = np.array(hist["px_true"])
            py = np.array(hist["py_true"])

            idx = np.where(t_arr <= t_target)[0]
            if len(idx) == 0:
                continue

            last = idx[-1] + 1
            px_plot = px[:last]
            py_plot = py[:last]

            if hist["success"]:
                ax.plot(px_plot, py_plot, color="blue", linewidth=2.0, alpha=0.85, zorder=5)
            else:
                ax.plot(px_plot, py_plot, "r--", linewidth=2.2, alpha=0.95, zorder=6)
                # if collision already happened by this panel time, show final shown point
                if t_arr[min(last - 1, len(t_arr)-1)] < t_target or len(t_arr) < int(SIM_TIME / DT) + 1:
                    ax.scatter(px_plot[-1], py_plot[-1], c="red", s=32, zorder=8)

        ax.set_xlim([-7, 7])
        ax.set_ylim([-2, 10])
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_title(f"t = {t_target} s", fontsize=16)

    fig.suptitle(
        f"10 Probabilistic Rollouts — Failures: {fail_count}/{NUM_PLOT_ROLLOUTS} "
        f"({100.0 * fail_count / NUM_PLOT_ROLLOUTS:.1f}%)\n"
        f"CP violations: {cp_viol_count}/{NUM_PLOT_ROLLOUTS}",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig("po_10_trajectories_three_timeslots.png", dpi=300)
    plt.show()

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("\n>>> Setting up 4D ODP grid...", flush=True)

    g = Grid(
        np.array([-7.0, -2.0, -4.0, -4.0]),
        np.array([7.0, 10.0, 4.0, 4.0]),
        4,
        np.array([30, 30, 15, 15]),
        []
    )

    tau = np.arange(0.0, T_MAX + 1e-9, DT)
    PX, PY, VX, VY = np.meshgrid(*g.grid_points, indexing="ij")
    compMethods = {"TargetSetMode": "maxCBF", "cbf_gamma": GAMMA}

    # -------------------------------------------------------------------------
    # STEP 1: BASELINE SOLVE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: BASELINE HJ/CBVF SOLVE")
    print("=" * 70)

    dist1_base = np.sqrt((PX - OBS1_C[0]) ** 2 + (PY - OBS1_C[1]) ** 2)
    dist2_base = np.sqrt((PX - OBS2_C[0]) ** 2 + (PY - OBS2_C[1]) ** 2)
    target_base = np.minimum(dist1_base - OBS1_R, dist2_base - OBS2_R).astype(np.float32)

    r_d_hat_base = L_EST_Y * (L_HY * V_BAR_DESIGN + V_BAR_DESIGN)
    sys_base = EstimatorHJModel4D(r_d_hat_max=r_d_hat_base)

    valfuncs_base = HJSolver(
        sys_base, g, target_base, tau, compMethods,
        saveAllTimeSteps=True, accuracy="low"
    )

    q_current = generate_calibration_rollouts(
        valfuncs_base, g, tau, r_d_hat_max=r_d_hat_base, num_rollouts=500
    )
    print(f"Initial baseline q_alpha = {q_current:.4f}")

    # -------------------------------------------------------------------------
    # STEP 2: FIXED-POINT CP CALIBRATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: FIXED-POINT CP CALIBRATION")
    print("=" * 70)

    MAX_ITERS = 3
    CONVERGENCE_TOL = 0.01
    is_converged = False

    def r_alpha_target(s, q):
        if L_EPS == 0:
            return q + W_BAR * s
        return q * np.exp(L_EPS * s) + (W_BAR / L_EPS) * (np.exp(L_EPS * s) - 1.0)

    for it in range(MAX_ITERS):
        print(f"\n--- Iteration {it + 1} ---")

        def time_varying_target(t_rev):
            t = T_MAX - t_rev
            r_err = r_alpha_target(t, q_current)

            eff_obs1_R = OBS1_R + r_err
            eff_obs2_R = OBS2_R + r_err

            dist1 = np.sqrt((PX - OBS1_C[0]) ** 2 + (PY - OBS1_C[1]) ** 2)
            dist2 = np.sqrt((PX - OBS2_C[0]) ** 2 + (PY - OBS2_C[1]) ** 2)

            target1 = dist1 - eff_obs1_R
            target2 = dist2 - eff_obs2_R
            return np.minimum(target1, target2).astype(np.float32)

        r_err_max = r_alpha_target(T_MAX, q_current)
        r_d_hat_max = L_EST_Y * (L_HY * r_err_max + V_BAR_DESIGN)

        sys_est = EstimatorHJModel4D(r_d_hat_max=r_d_hat_max)

        valfuncs_po = HJSolver(
            sys_est, g, time_varying_target, tau, compMethods,
            saveAllTimeSteps=True, accuracy="low"
        )

        q_new = generate_calibration_rollouts(
            valfuncs_po, g, tau, r_d_hat_max=r_d_hat_max, num_rollouts=500
        )

        print(f"q_current = {q_current:.4f} | q_new = {q_new:.4f}")

        if abs(q_new - q_current) <= CONVERGENCE_TOL:
            q_current = q_new
            is_converged = True
            print(f"[SUCCESS] Converged at q_alpha = {q_current:.4f}")
            break

        q_current = q_new

    if not is_converged:
        print("[WARNING] Strict fixed-point convergence not reached. Continuing with last iterate.")

    # -------------------------------------------------------------------------
    # STEP 3: FINAL SYNCHRONIZATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: FINAL SYNCHRONIZATION")
    print("=" * 70)

    def final_r_alpha(s):
        if L_EPS == 0:
            return q_current + W_BAR * s
        return q_current * np.exp(L_EPS * s) + (W_BAR / L_EPS) * (np.exp(L_EPS * s) - 1.0)

    def final_time_varying_target(t_rev):
        t = T_MAX - t_rev
        r_err = final_r_alpha(t)

        eff_obs1_R = OBS1_R + r_err
        eff_obs2_R = OBS2_R + r_err

        dist1 = np.sqrt((PX - OBS1_C[0]) ** 2 + (PY - OBS1_C[1]) ** 2)
        dist2 = np.sqrt((PX - OBS2_C[0]) ** 2 + (PY - OBS2_C[1]) ** 2)

        target1 = dist1 - eff_obs1_R
        target2 = dist2 - eff_obs2_R
        return np.minimum(target1, target2).astype(np.float32)

    r_err_max_final = final_r_alpha(T_MAX)
    r_d_hat_max_final = L_EST_Y * (L_HY * r_err_max_final + V_BAR_DESIGN)

    sys_est_final = EstimatorHJModel4D(r_d_hat_max=r_d_hat_max_final)

    valfuncs_po_final = HJSolver(
        sys_est_final, g, final_time_varying_target, tau, compMethods,
        saveAllTimeSteps=True, accuracy="low"
    )

    # -------------------------------------------------------------------------
    # STEP 4: VERIFY DESIGN-SIDE CONSISTENCY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: DESIGN-SIDE CONSISTENCY CHECK")
    print("=" * 70)

    q_check = generate_calibration_rollouts(
        valfuncs_po_final, g, tau, r_d_hat_max=r_d_hat_max_final, num_rollouts=500
    )

    print(f"Expected q_alpha = {q_current:.4f} | Verified q_alpha = {q_check:.4f}")

    if abs(q_check - q_current) > 0.02:
        print("[WARNING] Final value function is not perfectly synchronized with calibration.")
    else:
        print("[SUCCESS] Final design-side synchronization looks consistent.")

    # -------------------------------------------------------------------------
    # STEP 5: FINAL THREE-PANEL PLOT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: FINAL THREE-PANEL PLOT")
    print("=" * 70)

    plot_ten_rollouts_three_panels(
        valfuncs=valfuncs_po_final,
        g=g,
        tau=tau,
        r_d_hat_max=r_d_hat_max_final,
        q_alpha_final=q_current
    )

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY OF KEY NUMBERS")
    print("=" * 70)
    print(f"ALPHA                     : {ALPHA}")
    print(f"Final q_alpha             : {q_current:.4f}")
    print(f"r_err_max_final           : {r_err_max_final:.4f}")
    print(f"r_d_hat_max_final         : {r_d_hat_max_final:.4f}")
    print(f"Design D_MAG / V_BAR      : {D_MAG_DESIGN} / {V_BAR_DESIGN}")
    print(f"Deploy D_MAG / V_BAR      : {DEPLOY_D_MAG} / {DEPLOY_V_BAR}")
    print(f"Design observer gain      : {L_GAIN_DESIGN}")
    print(f"Deploy observer gain      : {DEPLOY_L_GAIN}")
    print("Output figure             : po_10_trajectories_three_timeslots.png")
    print("=" * 70)