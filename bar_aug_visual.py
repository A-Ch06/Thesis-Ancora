import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# === Paths ===
BASE_STATS_PATH = "D:/Thesis/outputs/output_statistics.json"
AUG_STATS_DIR = "D:/Thesis/outputs/aug_stats"
OUT_PLOTS_DIR = "D:/Thesis/outputs/aug_vs_base_plots"
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)

# === Nutrients and metrics ===
nutrients = ["calories", "fat", "carb", "protein"]
metrics = ["MAE", "MAE_%", "SMAPE"]

metric_colors = {
    "MAE": "#1f77b4",     # blue
    "MAE_%": "#ff7f0e",   # orange
    "SMAPE": "#2ca02c"    # green
}

legend_handles = [
    Patch(facecolor=metric_colors["MAE"], label="MAE"),
    Patch(facecolor=metric_colors["MAE_%"], label="MAE %"),
    Patch(facecolor=metric_colors["SMAPE"], label="SMAPE"),
    Patch(facecolor="gray", hatch="//", label="Baseline")
]

# === Load baseline ===
with open(BASE_STATS_PATH, "r") as f:
    base_stats = json.load(f)

# === Load augmented stats ===
aug_stats = {}
for fname in os.listdir(AUG_STATS_DIR):
    if fname.endswith(".json"):
        name = fname.replace("_stats.json", "")
        with open(os.path.join(AUG_STATS_DIR, fname)) as f:
            aug_stats[name] = json.load(f)

# Sort augmentation levels
aug_levels = sorted(aug_stats.keys())
all_levels = ["base"] + aug_levels 

# === Plot for each nutrient ===
for nutrient in nutrients:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_levels)) * 1.2
    width = 0.15
    gap = 0.15
    offsets = [-width - gap, 0, width + gap]  # 3 bars per group

    for i, metric in enumerate(metrics):
        values = []
        base_val = base_stats[f"{nutrient}_{metric}"]

        for level in all_levels:
            val = base_val if level == "base" else aug_stats[level][f"{nutrient}_{metric}"]
            values.append(val)

        # Horizontal baseline line for this metric
        ax.axhline(y=base_val, color=metric_colors[metric], linestyle='--', linewidth=1.2,
                   label=f"{metric} (base)", alpha=0.5, zorder=0)
        
        max_x_pos = x[-1] + offsets[-1] + 0.7
        ax.text(max_x_pos, base_val, f"{base_val:.1f}", color=metric_colors[metric],
        fontsize=8, va='center', ha='left', alpha=0.8)

        # Plot bars
        for j, val in enumerate(values):
            offset_x = x[j] + offsets[i]
            bar_color = "gray" if all_levels[j] == "base" else metric_colors[metric]
            hatch_style = "//" if all_levels[j] == "base" else None

            ax.bar(offset_x, val, width, color=bar_color, hatch=hatch_style,
                   zorder=2)

            # Show delta from base
            if all_levels[j] != "base":
                delta = val - base_val
                ax.text(offset_x, val + 1, f"{delta:+.1f}", ha="center", fontsize=8)

    ax.set_title(f"{nutrient.capitalize()} - Augmented vs Base\n(Bars = {', '.join(metrics)} | Labels = Δ from Base)")
    ax.set_ylabel("Value")
    ax.set_xlabel("Augmentation Intensity")
    ax.set_xticks(x)
    ax.set_xticklabels(all_levels, rotation=45)
    ax.legend(handles=legend_handles, title="Metrics")
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    save_path = os.path.join(OUT_PLOTS_DIR, f"{nutrient}_aug_vs_base.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")
