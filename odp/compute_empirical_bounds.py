import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ============================================================================
# SYSTEM PARAMETERS (Matching your main script)
# ============================================================================
D_MAG_THEORY = 0.1       # Theoretical max process noise
V_BAR_THEORY = 0.1      # Theoretical max measurement noise
L_GAIN = 2.0             # Observer gain
DT = 0.02                # Time step
T_MAX = 5.0              # Horizon
NUM_ROLLOUTS = 500
NUM_LIPSCHITZ_SAMPLES = 10000

print("\n" + "="*60)
print(">>> EXTRACTING EMPIRICAL CONSTANTS FOR PAPER <<<")
print("="*60)

# ============================================================================
# PART 1: EMPIRICAL LIPSCHITZ CONSTANTS (Monte Carlo Sampling)
# ============================================================================
print("1. Computing Empirical Lipschitz Constants...")

L_hy_ratios = []
L_est_y_ratios = []

for _ in range(NUM_LIPSCHITZ_SAMPLES):
    # Randomly sample states and measurements
    x1 = np.random.uniform(-10, 10, 4)
    x2 = np.random.uniform(-10, 10, 4)
    y1 = np.random.uniform(-10, 10, 2)
    y2 = np.random.uniform(-10, 10, 2)
    
    # --- Compute L_hy (Observation Lipschitz Constant) ---
    # h(x) simply extracts the position: [px, py]
    hx1 = x1[:2]
    hx2 = x2[:2]
    dist_x = np.linalg.norm(x1 - x2)
    if dist_x > 1e-6:
        L_hy_ratios.append(np.linalg.norm(hx1 - hx2) / dist_x)
        
    # --- Compute L_est_y (Estimator Sensitivity to Measurement) ---
    # Estimator update dynamics f_est(x_hat, y) based on your code:
    # x_hat_dot = [vx_hat + L*(y_x - px_hat), vy_hat + L*(y_y - py_hat), u_x, u_y]
    # We test how much the derivative changes when ONLY y changes
    x_hat = np.random.uniform(-10, 10, 4)
    
    f_est_1 = np.array([x_hat[2] + L_GAIN * (y1[0] - x_hat[0]), 
                        x_hat[3] + L_GAIN * (y1[1] - x_hat[1]), 0, 0])
    
    f_est_2 = np.array([x_hat[2] + L_GAIN * (y2[0] - x_hat[0]), 
                        x_hat[3] + L_GAIN * (y2[1] - x_hat[1]), 0, 0])
    
    dist_y = np.linalg.norm(y1 - y2)
    if dist_y > 1e-6:
        L_est_y_ratios.append(np.linalg.norm(f_est_1 - f_est_2) / dist_y)

L_HY_EMPIRICAL = np.max(L_hy_ratios)
L_EST_Y_EMPIRICAL = np.max(L_est_y_ratios)

# ============================================================================
# PART 2: EMPIRICAL TRAJECTORY BOUNDS (Rollouts)
# ============================================================================
print(f"2. Running {NUM_ROLLOUTS} rollouts to collect trajectory & noise data...")

steps = int(T_MAX / DT)
time_arr = np.linspace(0, T_MAX, steps)

error_trajectories = np.zeros((NUM_ROLLOUTS, steps))
process_noise_norms = []
measurement_noise_norms = []

for r in range(NUM_ROLLOUTS):
    x_t = np.array([0.0, 0.0, 0.0, 0.0])
    x_h = np.array([0.0, 0.0, 0.0, 0.0])
    
    for s in range(steps):
        # Sample noise
        d_x = np.random.uniform(-D_MAG_THEORY, D_MAG_THEORY)
        d_y = np.random.uniform(-D_MAG_THEORY, D_MAG_THEORY)
        v_x = np.random.uniform(-V_BAR_THEORY, V_BAR_THEORY)
        v_y = np.random.uniform(-V_BAR_THEORY, V_BAR_THEORY)
        
        process_noise_norms.append(np.linalg.norm([d_x, d_y]))
        measurement_noise_norms.append(np.linalg.norm([v_x, v_y]))
        
        # True Dynamics
        x_t[0] += x_t[2] * DT
        x_t[1] += x_t[3] * DT
        x_t[2] += d_x * DT
        x_t[3] += d_y * DT
        
        # Measurement & Estimator Dynamics
        y_meas_x = x_t[0] + v_x
        y_meas_y = x_t[1] + v_y
        
        x_h[0] += (x_h[2] + L_GAIN * (y_meas_x - x_h[0])) * DT
        x_h[1] += (x_h[3] + L_GAIN * (y_meas_y - x_h[1])) * DT
        
        # Record Error Distance
        error_trajectories[r, s] = np.linalg.norm(x_t - x_h)

D_MAG_EMPIRICAL = np.percentile(process_noise_norms, 99)
V_BAR_EMPIRICAL = np.percentile(measurement_noise_norms, 99)

# ============================================================================
# PART 3: TRAJECTORY REGRESSION (L_eps and w_bar)
# ============================================================================
print("3. Fitting Theoretical Bound to Empirical Trajectory Envelope...")
empirical_envelope = np.percentile(error_trajectories, 99, axis=0)

def error_bound_func(t, L_eps, w_bar):
    L_eps = np.maximum(L_eps, 1e-8)
    return (w_bar / L_eps) * (np.exp(L_eps * t) - 1.0)

popt, _ = curve_fit(error_bound_func, time_arr, empirical_envelope, 
                    p0=[0.1, 0.1], bounds=([0.0, 0.0], [5.0, 5.0]))

L_EPS_EMPIRICAL, W_BAR_EMPIRICAL = popt

# ============================================================================
# FINAL OUTPUT FOR PAPER
# ============================================================================
print("\n" + "="*60)
print(">>> FINAL EMPIRICAL CONSTANTS FOR MAIN CODE & PAPER <<<")
print("="*60)
print("1. Disturbance & Noise Bounds (99th Percentile)")
print(f"   D_MAG (Process Noise Bound)    : {D_MAG_EMPIRICAL:.4f}  (Theoretical max was {np.sqrt(2 * D_MAG_THEORY**2):.4f})")
print(f"   V_BAR (Measurement Noise Bound): {V_BAR_EMPIRICAL:.4f}  (Theoretical max was {np.sqrt(2 * V_BAR_THEORY**2):.4f})")

print("\n2. Lipschitz Sensitivities (Monte Carlo Max Ratio)")
print(f"   L_HY    (Observation mapping)  : {L_HY_EMPIRICAL:.4f}  (Expected exactly 1.0)")
print(f"   L_EST_Y (Estimator sensitivity): {L_EST_Y_EMPIRICAL:.4f}  (Expected exactly equal to L_GAIN = {L_GAIN})")

print("\n3. Trajectory Propagation Bounds (Nonlinear Regression)")
print(f"   L_EPS (Error Growth Rate)      : {L_EPS_EMPIRICAL:.4f}")
print(f"   W_BAR (Inst. Disturbance)      : {W_BAR_EMPIRICAL:.4f}")
print("="*60)
print("NOTE: Copy these exact values into the CONSTANTS section of po_double_integrator_cp.py!")

# (Optional) Plotting the verification
plt.figure(figsize=(8, 5))
for r in range(min(100, NUM_ROLLOUTS)): # Plot just 100 so it renders fast
    plt.plot(time_arr, error_trajectories[r], color='gray', alpha=0.05)
plt.plot(time_arr, empirical_envelope, color='blue', linewidth=2, label='99th Percentile Empirical Error')
fitted_curve = error_bound_func(time_arr, L_EPS_EMPIRICAL, W_BAR_EMPIRICAL)
plt.plot(time_arr, fitted_curve, color='red', linestyle='--', linewidth=2.5, 
         label=rf'Fitted Bound ($L_\varepsilon={L_EPS_EMPIRICAL:.4f}, \bar{{w}}={W_BAR_EMPIRICAL:.4f}$)')
plt.title('Fully Empirical Data-Driven Error Bounds', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Estimation Error Norm ||x - x_hat||', fontsize=12)
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig("empirical_error_fit_comprehensive.png", dpi=300)