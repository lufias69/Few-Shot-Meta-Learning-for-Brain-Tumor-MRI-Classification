"""
Specialized Visualizations for PAPER_4_HASIL_PEMBAHASAN.md
===========================================================

Creates publication-quality figures specifically for Results & Discussion section:
1. Training progression with detailed annotations
2. Learning phases breakdown (3 phases with LR)
3. Confusion matrix with clinical annotations and error patterns
4. Per-class detailed performance with error analysis
5. Baseline comparison with statistical significance
6. Clinical decision support metrics

Author: Research Team
Date: December 4, 2025
Purpose: SINTA 1 Journal Manuscript - Section IV (Hasil dan Pembahasan)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Color palette
COLORS = {
    'glioma': '#E63946',
    'meningioma': '#457B9D',
    'notumor': '#06A77D',
    'pituitary': '#F77F00',
    'train': '#2E86AB',
    'val': '#A23B72',
    'phase1': '#FF6B6B',
    'phase2': '#4ECDC4',
    'phase3': '#45B7D1'
}

print("=" * 80)
print("GENERATING PAPER 4 VISUALIZATIONS")
print("=" * 80)

# ============================================================================
# DATA LOADING
# ============================================================================
print("\n[INFO] Loading data from paper...")

# Training trajectory (20 checkpoints)
episodes = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
                     550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])

val_acc = np.array([57.76, 68.16, 72.40, 77.61, 75.45, 71.51, 84.96, 76.76, 84.34, 86.66,
                    84.63, 88.61, 87.49, 86.15, 83.93, 89.36, 88.60, 88.01, 91.14, 91.57])

val_loss = np.array([1.1544, 0.7957, 0.7675, 0.7111, 0.7245, 0.7766, 0.5472, 0.7039, 0.5735, 0.5274,
                     0.5606, 0.4662, 0.5028, 0.5358, 0.5483, 0.4460, 0.4720, 0.4726, 0.4099, 0.3983])

train_acc = np.array([65, 87.5, 87.5, 80, 92.5, 90, 85, 92.5, 85, 95,
                      87.5, 92.5, 92.5, 92.5, 95, 95, 95, 90, 92.5, 92.5])

train_loss = np.array([0.9844, 0.5625, 0.5312, 0.6406, 0.4219, 0.4687, 0.5469, 0.4219, 0.5156, 0.3750,
                       0.5156, 0.4531, 0.4219, 0.4531, 0.4062, 0.3281, 0.3438, 0.4531, 0.4375, 0.4062])

# Test results
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
cm = np.array([
    [855, 54, 36, 55],
    [96, 736, 80, 88],
    [43, 93, 818, 46],
    [64, 80, 50, 806]
])

precision = np.array([80.81, 76.43, 83.13, 81.01])
recall = np.array([85.50, 73.60, 81.80, 80.60])
f1_score = np.array([83.09, 74.99, 82.46, 80.80])

# Baseline comparisons
methods = ['QSVM\n(Pure)', 'Hybrid\nQNN', 'Few-Shot\nMeta-Learning']
accuracies = [22.5, 40.6, 80.38]
std_devs = [3.2, 4.5, 5.02]

print("✓ Data loaded successfully!")

# ============================================================================
# FIGURE 1: Training Progression with Learning Phases
# ============================================================================
print("\n[1/6] Creating Training Progression Analysis...")

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

# Main plot: Accuracy with phase annotations
ax_main = fig.add_subplot(gs[0:2, :])

# Phase backgrounds
phases = [
    (0, 200, COLORS['phase1'], 'Phase 1\nRapid Learning\nLR=0.001'),
    (200, 400, COLORS['phase2'], 'Phase 2\nPlateau & Recovery\nLR=0.001→0.0005'),
    (400, 1000, COLORS['phase3'], 'Phase 3\nRefinement\nLR=0.0005→0.00025')
]

for start, end, color, label in phases:
    ax_main.axvspan(start, end, alpha=0.1, color=color)
    mid = (start + end) / 2
    ax_main.text(mid, 96, label, ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3))

# Plot lines
ax_main.plot(episodes, train_acc, 'o-', color=COLORS['train'], linewidth=2.5,
            markersize=8, label='Training Accuracy', alpha=0.9, markeredgecolor='black', markeredgewidth=0.5)
ax_main.plot(episodes, val_acc, 's-', color=COLORS['val'], linewidth=2.5,
            markersize=8, label='Validation Accuracy', alpha=0.9, markeredgecolor='black', markeredgewidth=0.5)

# Key milestones
milestones = [
    (50, val_acc[0], '57.76%\nInitial', 'bottom'),
    (200, val_acc[3], '77.61%\n+19.85%', 'top'),
    (300, val_acc[5], '71.51%\nPlateau', 'bottom'),
    (350, val_acc[6], '84.96%\nBreakthrough\n+13.45%', 'top'),
    (1000, val_acc[-1], '91.57%\nFINAL BEST', 'top')
]

for ep, acc, text, pos in milestones:
    idx = np.where(episodes == ep)[0][0]
    if pos == 'top':
        xytext = (ep, acc + 8)
        va = 'bottom'
    else:
        xytext = (ep, acc - 8)
        va = 'top'
    
    ax_main.annotate(text, xy=(ep, acc), xytext=xytext,
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                    fontsize=9, ha='center', va=va, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='darkred', alpha=0.9))

# Best model marker
best_idx = np.argmax(val_acc)
ax_main.plot(episodes[best_idx], val_acc[best_idx], '*', color='gold',
            markersize=30, markeredgecolor='darkred', markeredgewidth=3, zorder=10)

ax_main.set_xlabel('Training Episode', fontweight='bold', fontsize=12)
ax_main.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
ax_main.set_title('Training and Validation Accuracy Progression (1,000 Episodes)', 
                 fontweight='bold', fontsize=14, pad=15)
ax_main.grid(True, alpha=0.3, linestyle='--')
ax_main.legend(loc='lower right', framealpha=0.95, fontsize=10)
ax_main.set_ylim(50, 102)
ax_main.set_xlim(-50, 1050)

# Bottom left: Phase statistics
ax_stat = fig.add_subplot(gs[2, 0])
phase_names = ['Phase 1\n(Ep 1-200)', 'Phase 2\n(Ep 200-400)', 'Phase 3\n(Ep 400-1000)']
phase_improve = [19.85, 7.35, 14.81]  # Improvements
phase_colors = [COLORS['phase1'], COLORS['phase2'], COLORS['phase3']]

bars = ax_stat.bar(phase_names, phase_improve, color=phase_colors, alpha=0.7,
                  edgecolor='black', linewidth=2)
for i, (bar, val) in enumerate(zip(bars, phase_improve)):
    ax_stat.text(i, val + 0.5, f'+{val:.2f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

ax_stat.set_ylabel('Accuracy Improvement (%)', fontweight='bold')
ax_stat.set_title('Per-Phase Learning Progress', fontweight='bold', fontsize=11)
ax_stat.grid(True, alpha=0.3, axis='y')
ax_stat.set_ylim(0, 25)

# Bottom right: Loss curves
ax_loss = fig.add_subplot(gs[2, 1])
ax_loss.plot(episodes, train_loss, 'o-', color=COLORS['train'], linewidth=2,
            markersize=6, label='Training Loss', alpha=0.8)
ax_loss.plot(episodes, val_loss, 's-', color=COLORS['val'], linewidth=2,
            markersize=6, label='Validation Loss', alpha=0.8)

# Mark final convergence
ax_loss.plot(episodes[-1], train_loss[-1], 'o', color='green', markersize=15,
            markeredgecolor='black', markeredgewidth=2)
ax_loss.plot(episodes[-1], val_loss[-1], 's', color='green', markersize=15,
            markeredgecolor='black', markeredgewidth=2)
ax_loss.text(episodes[-1] + 30, (train_loss[-1] + val_loss[-1])/2,
            f'Final:\nTrain: 0.406\nVal: 0.398\nGap: -0.008',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

ax_loss.set_xlabel('Training Episode', fontweight='bold')
ax_loss.set_ylabel('Loss (Cross-Entropy)', fontweight='bold')
ax_loss.set_title('Loss Convergence Analysis', fontweight='bold', fontsize=11)
ax_loss.grid(True, alpha=0.3, linestyle='--')
ax_loss.legend(loc='upper right', framealpha=0.95)

plt.savefig('Paper4_Fig1_Training_Progression.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: Paper4_Fig1_Training_Progression.png")

# ============================================================================
# FIGURE 2: Detailed Confusion Matrix with Error Analysis
# ============================================================================
print("\n[2/6] Creating Detailed Confusion Matrix...")

fig = plt.figure(figsize=(16, 7))
gs = GridSpec(1, 2, figure=fig, wspace=0.3)

# Left: Confusion matrix with annotations
ax1 = fig.add_subplot(gs[0])
cm_display = cm.copy()

sns.heatmap(cm_display, annot=False, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
            xticklabels=classes, yticklabels=classes, ax=ax1,
            linewidths=2, linecolor='white', square=True, vmin=0, vmax=1000)

# Custom annotations
for i in range(4):
    for j in range(4):
        count = cm[i, j]
        pct = (count / cm[i, :].sum()) * 100
        
        if i == j:  # Diagonal - correct
            color = 'white'
            weight = 'bold'
            size = 11
            text = f'{count}\n({pct:.1f}%)\n✓ CORRECT'
        else:  # Off-diagonal - errors
            color = 'red' if pct > 8 else 'darkblue'
            weight = 'bold' if pct > 8 else 'normal'
            size = 10 if pct > 8 else 9
            text = f'{count}\n({pct:.1f}%)'
            
            # Highlight major confusions
            if pct > 8:
                rect = Rectangle((j, i), 1, 1, fill=False, edgecolor='red',
                               linewidth=4, linestyle='--')
                ax1.add_patch(rect)
        
        ax1.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                color=color, fontsize=size, fontweight=weight)

ax1.set_xlabel('Predicted Class', fontweight='bold', fontsize=12)
ax1.set_ylabel('True Class', fontweight='bold', fontsize=12)
ax1.set_title('Confusion Matrix (4,000 Predictions)\nDiagonal: 3,215 Correct (80.38%) | Off-Diagonal: 785 Errors (19.62%)',
             fontweight='bold', fontsize=12, pad=15)

# Right: Error analysis breakdown
ax2 = fig.add_subplot(gs[1])

# Major confusions (>8%)
major_errors = [
    ('Meningioma\n→ Glioma', 96, 9.6, COLORS['meningioma']),
    ('Meningioma\n→ No Tumor', 93, 9.3, COLORS['meningioma']),
    ('Meningioma\n→ Pituitary', 88, 8.8, COLORS['meningioma'])
]

y_pos = np.arange(len(major_errors))
error_counts = [e[1] for e in major_errors]
error_pcts = [e[2] for e in major_errors]
error_colors = [e[3] for e in major_errors]

bars = ax2.barh(y_pos, error_counts, color=error_colors, alpha=0.7,
               edgecolor='black', linewidth=2)

# Add percentage labels
for i, (name, count, pct, color) in enumerate(major_errors):
    ax2.text(count + 3, i, f'{count} errors\n({pct:.1f}%)',
            va='center', fontsize=10, fontweight='bold')

ax2.set_yticks(y_pos)
ax2.set_yticklabels([e[0] for e in major_errors], fontsize=10)
ax2.set_xlabel('Number of Misclassifications', fontweight='bold', fontsize=11)
ax2.set_title('Major Confusion Patterns (>8% Error Rate)\nAll from Meningioma Class',
             fontweight='bold', fontsize=12, pad=15)
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_xlim(0, 110)

# Add clinical note
note_text = (
    "Clinical Interpretation:\n\n"
    "• Meningioma most challenging (26.4% error rate)\n"
    "• Confused with all other classes\n"
    "• Well-circumscribed borders overlap with pituitary\n"
    "• Homogeneous intensity mimics no-tumor\n"
    "• Occasional infiltrative appearance similar to glioma\n\n"
    "→ Requires additional clinical context\n"
    "   (multi-sequence MRI, radiologist review)"
)
ax2.text(1.02, 0.5, note_text, transform=ax2.transAxes, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', edgecolor='orange', linewidth=2),
        verticalalignment='center')

plt.savefig('Paper4_Fig2_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: Paper4_Fig2_Confusion_Matrix.png")

# ============================================================================
# FIGURE 3: Per-Class Performance Detailed Analysis
# ============================================================================
print("\n[3/6] Creating Per-Class Performance Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Panel A: Precision, Recall, F1 comparison
ax1 = axes[0, 0]
x = np.arange(len(classes))
width = 0.25

bars1 = ax1.bar(x - width, precision, width, label='Precision',
               color=COLORS['glioma'], alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x, recall, width, label='Recall',
               color=COLORS['meningioma'], alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax1.bar(x + width, f1_score, width, label='F1-Score',
               color=COLORS['notumor'], alpha=0.8, edgecolor='black', linewidth=1.5)

# Value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Reference lines
ax1.axhline(y=80, color='green', linestyle='--', linewidth=2, alpha=0.5, label='80% Target')
ax1.axhline(y=70, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='70% Minimum')

ax1.set_ylabel('Performance (%)', fontweight='bold', fontsize=11)
ax1.set_title('(A) Per-Class Metrics Comparison', fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(classes, fontsize=10)
ax1.legend(loc='lower left', framealpha=0.95)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 100)

# Panel B: Error rate breakdown
ax2 = axes[0, 1]
error_rates = 100 - recall
correct_rates = recall

bars_correct = ax2.barh(classes, correct_rates, 
                        color=[COLORS['glioma'], COLORS['meningioma'], 
                              COLORS['notumor'], COLORS['pituitary']],
                        alpha=0.8, edgecolor='black', linewidth=1.5, label='Correct')
bars_error = ax2.barh(classes, error_rates, left=correct_rates,
                     color='lightcoral', alpha=0.6, edgecolor='red', 
                     linewidth=1.5, linestyle='--', label='Errors')

# Labels
for i, (correct, error) in enumerate(zip(correct_rates, error_rates)):
    ax2.text(correct/2, i, f'{correct:.1f}%', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    ax2.text(correct + error/2, i, f'{error:.1f}%\nerror', ha='center', va='center',
            fontsize=8, fontweight='bold')

ax2.set_xlabel('Percentage (%)', fontweight='bold', fontsize=11)
ax2.set_title('(B) Recall vs Error Rate Distribution', fontweight='bold', fontsize=12)
ax2.legend(loc='lower right', framealpha=0.95)
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_xlim(0, 100)

# Panel C: Support vs Performance
ax3 = axes[1, 0]
support = cm.sum(axis=1)
accuracy_per_class = np.diag(cm) / support * 100

ax3_twin = ax3.twinx()

bars = ax3.bar(classes, support, 
              color=[COLORS['glioma'], COLORS['meningioma'], 
                    COLORS['notumor'], COLORS['pituitary']],
              alpha=0.5, edgecolor='black', linewidth=1.5, label='Sample Count')
line = ax3_twin.plot(classes, accuracy_per_class, 'D-', color='darkred',
                     linewidth=3, markersize=12, markeredgecolor='black',
                     markeredgewidth=2, label='Accuracy (%)')

# Labels
for i, (s, a) in enumerate(zip(support, accuracy_per_class)):
    ax3.text(i, s/2, f'n={s}', ha='center', va='center',
            fontsize=11, fontweight='bold', color='darkblue')
    ax3_twin.text(i, a + 2, f'{a:.1f}%', ha='center', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

ax3.set_ylabel('Number of Test Samples', fontweight='bold', fontsize=11, color='blue')
ax3_twin.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=11, color='darkred')
ax3.set_title('(C) Sample Distribution vs Accuracy', fontweight='bold', fontsize=12)
ax3.tick_params(axis='y', labelcolor='blue')
ax3_twin.tick_params(axis='y', labelcolor='darkred')
ax3.grid(True, alpha=0.3)
ax3_twin.set_ylim(70, 90)

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.95)

# Panel D: Class difficulty ranking
ax4 = axes[1, 1]
difficulty_scores = 100 - f1_score
class_difficulty = list(zip(classes, f1_score, difficulty_scores,
                            [COLORS['glioma'], COLORS['meningioma'], 
                             COLORS['notumor'], COLORS['pituitary']]))
class_difficulty.sort(key=lambda x: x[2], reverse=True)

y_pos = np.arange(len(class_difficulty))
difficulties = [x[2] for x in class_difficulty]
colors_sorted = [x[3] for x in class_difficulty]
labels_sorted = [f"{x[0]}\n(F1={x[1]:.1f}%)" for x in class_difficulty]

bars = ax4.barh(y_pos, difficulties, color=colors_sorted, alpha=0.7,
               edgecolor='black', linewidth=1.5)

# Add difficulty labels
difficulty_labels = ['MOST DIFFICULT', 'CHALLENGING', 'MODERATE', 'EASIEST']
for i, (bar, label) in enumerate(zip(bars, difficulty_labels)):
    width = bar.get_width()
    ax4.text(width + 0.5, i, f'{width:.1f}%\n{label}',
            va='center', fontsize=9, fontweight='bold')

ax4.set_yticks(y_pos)
ax4.set_yticklabels(labels_sorted, fontsize=10)
ax4.set_xlabel('Difficulty Score (100 - F1)', fontweight='bold', fontsize=11)
ax4.set_title('(D) Class Difficulty Ranking', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3, axis='x')
ax4.set_xlim(0, 30)
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig('Paper4_Fig3_PerClass_Analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: Paper4_Fig3_PerClass_Analysis.png")

# ============================================================================
# FIGURE 4: Baseline Comparison with Statistical Significance
# ============================================================================
print("\n[4/6] Creating Baseline Comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Accuracy comparison with error bars
x_pos = np.arange(len(methods))
colors_methods = ['#d62728', '#ff7f0e', '#2ca02c']

bars = ax1.bar(x_pos, accuracies, color=colors_methods, alpha=0.7,
              edgecolor='black', linewidth=2, width=0.6)
ax1.errorbar(x_pos, accuracies, yerr=std_devs, fmt='none',
            ecolor='black', capsize=10, capthick=2)

# Statistical significance markers
ax1.plot([0, 2], [88, 88], 'k-', linewidth=2)
ax1.text(1, 90, '***', ha='center', fontsize=24, fontweight='bold')
ax1.text(1, 93, 'p < 0.001\nCohen\'s d = 12.8', ha='center', fontsize=9, 
        style='italic', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Value labels with improvement
for i, (acc, std) in enumerate(zip(accuracies, std_devs)):
    ax1.text(i, acc + std + 4, f'{acc:.2f}%\n± {std:.2f}%',
            ha='center', fontsize=10, fontweight='bold')
    
    if i > 0:
        improvement = acc - accuracies[i-1]
        ax1.annotate(f'+{improvement:.1f}%', xy=(i, acc/2), fontsize=9,
                    color='green', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

ax1.set_ylabel('Test Accuracy (%)', fontweight='bold', fontsize=12)
ax1.set_title('(A) Method Comparison with Statistical Significance',
             fontweight='bold', fontsize=12)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(methods, fontsize=10)
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3, axis='y')

# Reference line
ax1.axhline(y=80, color='blue', linestyle='--', linewidth=2, alpha=0.5,
           label='Clinical Target (80%)')
ax1.legend(loc='upper left', framealpha=0.95)

# Panel B: Improvement breakdown
ax2_data = {
    'Metric': ['Accuracy', 'vs QSVM', 'vs Hybrid'],
    'QSVM': [22.5, 0, 18.1],
    'Hybrid QNN': [40.6, 18.1, 0],
    'Few-Shot ML': [80.38, 57.88, 39.78]
}

x = np.arange(len(ax2_data['Metric']))
width = 0.25

bars1 = ax2.bar(x - width, ax2_data['QSVM'], width, label='QSVM',
               color='#d62728', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x, ax2_data['Hybrid QNN'], width, label='Hybrid QNN',
               color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=1.5)
bars3 = ax2.bar(x + width, ax2_data['Few-Shot ML'], width, label='Few-Shot ML',
               color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=1.5)

# Value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

ax2.set_ylabel('Value (%)', fontweight='bold', fontsize=11)
ax2.set_title('(B) Performance Metrics Breakdown', fontweight='bold', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(ax2_data['Metric'], fontsize=10)
ax2.legend(loc='upper left', framealpha=0.95)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 90)

plt.tight_layout()
plt.savefig('Paper4_Fig4_Baseline_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: Paper4_Fig4_Baseline_Comparison.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PAPER 4 VISUALIZATIONS COMPLETE!")
print("=" * 80)
print("\nGenerated Figures for Section IV (Hasil dan Pembahasan):")
print("  1. Paper4_Fig1_Training_Progression.png    - Training phases & milestones")
print("  2. Paper4_Fig2_Confusion_Matrix.png        - Detailed confusion & error patterns")
print("  3. Paper4_Fig3_PerClass_Analysis.png       - Per-class metrics (4-panel)")
print("  4. Paper4_Fig4_Baseline_Comparison.png     - Statistical comparison")
print("\nAll figures: 300 DPI, publication-ready for SINTA 1 journal")
print("=" * 80)
