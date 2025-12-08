"""
Create Figure 3: The Dimensional Validity Bound

Shows the U-shaped relationship between multimorbidity, D_eff, and AUC.
Highlights the "zone of maximum entropy" where Bayesian inference fails.

Styling harmonized with plot_npj_style.py (Figures 1-2).
"""

import matplotlib.pyplot as plt
import numpy as np

# NPJ-style settings (matched to Figures 1-2)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# Color palette (Nature-style, matched to Figures 1-2)
COLORS = {
    'primary': '#2166AC',      # Blue (D_eff)
    'secondary': '#B2182B',    # Red (AUC)
    'gray': '#666666',
    'zone': '#E8E8E8',
}

# Data from MIMIC-IV analysis
strata = ['Low', 'Moderate', 'High']
multimorbidity = [6, 12, 21]  # median diagnoses
d_eff = [44.3, 37.5, 29.5]
auc = [0.867, 0.835, 0.859]
auc_ci_low = [0.858, 0.832, 0.851]
auc_ci_high = [0.876, 0.839, 0.867]

# Create figure
fig, ax1 = plt.subplots(figsize=(10, 5.5))

# Plot D_eff on left axis
color1 = COLORS['primary']
ax1.set_xlabel('Multimorbidity Burden (Median ICD Codes)', fontsize=12)
ax1.set_ylabel('Effective Dimensionality ($D_{eff}$)', color=color1, fontsize=12)
line1 = ax1.plot(multimorbidity, d_eff, 'o-', color=color1, linewidth=2,
                  markersize=10, label='$D_{eff}$')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(25, 50)

# Add D_eff annotations
for i, (x, y) in enumerate(zip(multimorbidity, d_eff)):
    ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                 xytext=(0, 10), ha='center', color=color1, fontsize=10)

# Create second y-axis for AUC
ax2 = ax1.twinx()
color2 = COLORS['secondary']

ax2.set_ylabel('AUC-ROC', color=color2, fontsize=12)
line2 = ax2.errorbar(multimorbidity, auc,
                      yerr=[np.array(auc) - np.array(auc_ci_low),
                            np.array(auc_ci_high) - np.array(auc)],
                      fmt='s-', color=color2, linewidth=2, markersize=10,
                      capsize=5, label='AUC')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0.82, 0.90)

# Add AUC annotations - offset to the right to avoid line collision
auc_offsets = [(15, 5), (15, -15), (-20, 5)]  # custom offsets per point
for i, (x, y) in enumerate(zip(multimorbidity, auc)):
    ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                 xytext=auc_offsets[i], ha='left' if auc_offsets[i][0] > 0 else 'right',
                 color=color2, fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='none', alpha=0.8))

# Highlight the "Zone of Maximum Entropy"
ax1.axvspan(9, 15, alpha=0.4, color=COLORS['zone'], lw=0, zorder=0)
ax1.annotate('Zone of Maximum Entropy\n(Posterior Instability)',
             xy=(12, 46), ha='center', fontsize=10,
             style='italic', color=COLORS['gray'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='none', alpha=0.9))

# Add stratum labels
for i, (x, label) in enumerate(zip(multimorbidity, strata)):
    ax1.annotate(label, (x, 26), ha='center', fontsize=10, fontweight='bold')

# Add interpretation arrows
ax1.annotate('', xy=(6, 44.3), xytext=(21, 29.5),
            arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1.5,
                          connectionstyle='arc3,rad=-0.2', ls='--'))
ax1.annotate('Manifold\nCollapse', xy=(14, 34), fontsize=9, color=COLORS['gray'],
            ha='center', style='italic')

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + [line2], labels1 + labels2, loc='upper right', framealpha=0.9)

# Title
plt.title('The Dimensional Validity Bound in MIMIC-IV (N=425,216)', fontsize=14, pad=20)

# Grid
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks([6, 12, 21])

plt.tight_layout()

# Save
plt.savefig('../figures/fig3_validity_bound.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('../figures/fig3_validity_bound.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print("Figure saved to figures/fig3_validity_bound.png")
