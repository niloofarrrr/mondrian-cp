import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Load the three result files
# ---------------------------------------------------------------------------
paths = {
    "AlignedModel": "plots/aligned_no_mismatch/experiment_results_aligned_no_mismatch.json",
    "EasyMismatch":  "plots/easy_mismatch_matched_policy/experiment_results_easy_mismatch.json",
    "HardMismatch":  "plots/hard_mismatch_matched_policy/experiment_results_hard_mismatch.json",
}

combined = {}
for variant_name, path in paths.items():
    with open(path) as f:
        data = json.load(f)
    # correct key is results_mean_std_across_seeds
    inner_dict = data["results_mean_std_across_seeds"]
    # the inner dict has one key equal to the variant name
    inner = list(inner_dict.values())[0]
    combined[variant_name] = {
        "nominal": inner["nominal"],
        "cbvf":    inner["cbvf"],
        "cp":      inner["cp"],
    }

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
colors  = {"nominal": "#e15759", "cbvf": "#4e79a7", "cp": "#59a14f"}
hatches = {"nominal": "",        "cbvf": "//",       "cp": ".."}
labels  = {"nominal": "nominal", "cbvf": "cbvf",     "cp": "cp"}
methods = ["nominal", "cbvf", "cp"]
variant_names = list(combined.keys())
x     = np.arange(len(variant_names))
width = 0.22

out_dir = Path("plots/combined_plots")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def save_grouped(metric, ylabel, title, ylim=None, fname=None):
    plt.figure(figsize=(9, 4.5))
    for j, m in enumerate(methods):
        vals = [combined[v][m].get(metric, 0.0) for v in variant_names]
        errs = [combined[v][m].get(f"{metric}_std_across_seeds", 0.0)
                for v in variant_names]
        use_errs = any(e > 0 for e in errs)
        plt.bar(
            x + (j - 1) * width, vals, width=width,
            yerr=errs if use_errs else None,
            capsize=3 if use_errs else 0,
            label=labels[m], color=colors[m], alpha=0.88,
            edgecolor="black", linewidth=0.9, hatch=hatches[m],
        )
    plt.xticks(x, variant_names)
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    elif "rate" in metric:
        plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / (fname or f"{metric}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")

# ---------------------------------------------------------------------------
# Figure 1: safety_comparison — side-by-side unsafe_rate and goal_rate
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (metric, ylabel) in zip(axes, [
    ("unsafe_rate", "Unsafe episode rate"),
    ("goal_rate",   "Goal-reaching rate"),
]):
    for j, m in enumerate(methods):
        vals = [combined[v][m].get(metric, 0.0) for v in variant_names]
        errs = [combined[v][m].get(f"{metric}_std_across_seeds", 0.0)
                for v in variant_names]
        use_errs = any(e > 0 for e in errs)
        ax.bar(
            x + (j - 1) * width, vals, width=width,
            yerr=errs if use_errs else None,
            capsize=3 if use_errs else 0,
            label=labels[m], color=colors[m], alpha=0.88,
            edgecolor="black", linewidth=0.9, hatch=hatches[m],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(variant_names)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.legend()

fig.suptitle("Safety comparison: Nominal vs CBVF vs Conformal-CBVF shield")
plt.tight_layout()
out_path = out_dir / "safety_comparison.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved {out_path}")

# ---------------------------------------------------------------------------
# Figure 2: interventions
# ---------------------------------------------------------------------------
save_grouped(
    metric="mean_interventions",
    ylabel="Mean interventions per rollout",
    title="Shield interventions per rollout",
    fname="interventions.png",
)

# ---------------------------------------------------------------------------
# Figure 3: mean return
# ---------------------------------------------------------------------------
save_grouped(
    metric="mean_return",
    ylabel="Mean rollout return",
    title="Mean rollout return by scenario",
    fname="mean_return.png",
)

# ---------------------------------------------------------------------------
# Figure 4: goal rate standalone
# ---------------------------------------------------------------------------
save_grouped(
    metric="goal_rate",
    ylabel="Goal-reaching rate",
    title="Goal-reaching rate",
    ylim=(0.0, 1.0),
    fname="goal_rate.png",
)

# ---------------------------------------------------------------------------
# Figure 5: unsafe rate standalone
# ---------------------------------------------------------------------------
save_grouped(
    metric="unsafe_rate",
    ylabel="Unsafe episode rate",
    title="Unsafe episode rate",
    ylim=(0.0, 1.0),
    fname="unsafe_rate.png",
)

print(f"\nAll combined plots saved to {out_dir}/")