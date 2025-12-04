"""
Generate High-Quality Publication Figures for Few-Shot Meta-Learning Paper
===========================================================================

Creates professional visualizations for SINTA 1 journal publication:
1. Training & Validation Curves (dual-axis, publication quality)
2. Confusion Matrix Heatmap (annotated, normalized)
3. Per-Class Performance Comparison (bar charts with error bars)
4. Model Architecture Diagram (visual representation)
5. Episode Structure Illustration (N-way K-shot visualization)
6. Clinical Decision Support Workflow (flowchart)

Author: Research Team
Date: December 4, 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# Set publication-quality style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Color palette (colorblind-friendly)
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9896',
    'info': '#9467bd',
    'light': '#e0e0e0',
    'dark': '#333333'
}

print("=" * 80)
print("GENERATING PUBLICATION-QUALITY FIGURES")
print("=" * 80)

# ============================================================================
# FIGURE 1: Training & Validation Learning Curves (Enhanced)
# ============================================================================
print("\n[1/6] Generating Training & Validation Curves...")

# Training data (from actual results)
episodes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 
            550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
val_acc = [57.76, 68.16, 72.40, 77.61, 75.45, 71.51, 84.96, 76.76, 84.34, 86.66,
           84.63, 88.61, 87.49, 86.15, 83.93, 89.36, 88.60, 88.01, 91.14, 91.57]
val_loss = [1.1544, 0.7957, 0.7675, 0.7111, 0.7245, 0.7766, 0.5472, 0.7039, 0.5735, 0.5274,
            0.5606, 0.4662, 0.5028, 0.5358, 0.5483, 0.4460, 0.4720, 0.4726, 0.4099, 0.3983]

train_acc = [65, 87.5, 87.5, 80, 92.5, 90, 85, 92.5, 85, 95,
             87.5, 92.5, 92.5, 92.5, 95, 95, 95, 90, 92.5, 92.5]
train_loss = [0.9844, 0.5625, 0.5312, 0.6406, 0.4219, 0.4687, 0.5469, 0.4219, 0.5156, 0.3750,
              0.5156, 0.4531, 0.4219, 0.4531, 0.4062, 0.3281, 0.3438, 0.4531, 0.4375, 0.4062]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Accuracy
ax1.plot(episodes, train_acc, 'o-', color=COLORS['primary'], linewidth=2, 
         markersize=6, label='Training Accuracy', alpha=0.8)
ax1.plot(episodes, val_acc, 's-', color=COLORS['secondary'], linewidth=2, 
         markersize=6, label='Validation Accuracy', alpha=0.8)

# Mark best model
best_idx = val_acc.index(max(val_acc))
ax1.plot(episodes[best_idx], val_acc[best_idx], '*', color=COLORS['danger'], 
         markersize=20, label=f'Best Model (Ep {episodes[best_idx]})', zorder=5)

# Add annotations for key milestones
milestones = [
    (200, 77.61, 'LR=0.001'),
    (350, 84.96, 'LR=0.0005\nBreakthrough'),
    (600, 88.61, 'LR=0.00025'),
    (1000, 91.57, 'Final Best\n91.57%')
]
for ep, acc, text in milestones:
    idx = episodes.index(ep)
    ax1.annotate(text, xy=(ep, acc), xytext=(ep+50, acc-3),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                fontsize=8, ha='left', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', edgecolor='gray', alpha=0.8))

ax1.set_xlabel('Training Episode', fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontweight='bold')
ax1.set_title('(A) Training & Validation Accuracy Progression', fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='lower right', framealpha=0.9)
ax1.set_xlim(0, 1050)
ax1.set_ylim(50, 100)

# Subplot 2: Loss
ax2.plot(episodes, train_loss, 'o-', color=COLORS['primary'], linewidth=2, 
         markersize=6, label='Training Loss', alpha=0.8)
ax2.plot(episodes, val_loss, 's-', color=COLORS['secondary'], linewidth=2, 
         markersize=6, label='Validation Loss', alpha=0.8)

# Mark best model (lowest val loss)
best_loss_idx = val_loss.index(min(val_loss))
ax2.plot(episodes[best_loss_idx], val_loss[best_loss_idx], '*', 
         color=COLORS['danger'], markersize=20, label=f'Lowest Loss (Ep {episodes[best_loss_idx]})', zorder=5)

ax2.set_xlabel('Training Episode', fontweight='bold')
ax2.set_ylabel('Loss (Cross-Entropy)', fontweight='bold')
ax2.set_title('(B) Training & Validation Loss Reduction', fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', framealpha=0.9)
ax2.set_xlim(0, 1050)
ax2.set_ylim(0, 1.3)

plt.tight_layout()
plt.savefig('Figure_1_Training_Curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: Figure_1_Training_Curves.png")

# ============================================================================
# FIGURE 2: Confusion Matrix with Clinical Interpretation
# ============================================================================
print("\n[2/6] Generating Enhanced Confusion Matrix...")

# Confusion matrix data (from test results)
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
cm = np.array([
    [855, 54, 36, 55],
    [96, 736, 80, 88],
    [43, 93, 818, 46],
    [64, 80, 50, 806]
])

# Normalize by row (recall view)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Absolute counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
            xticklabels=classes, yticklabels=classes, ax=ax1, 
            linewidths=1, linecolor='white', square=True)
ax1.set_xlabel('Predicted Class', fontweight='bold')
ax1.set_ylabel('True Class', fontweight='bold')
ax1.set_title('(A) Confusion Matrix (Absolute Counts)', fontweight='bold', pad=10)

# Add diagonal highlights
for i in range(len(classes)):
    ax1.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='green', lw=3))

# Subplot 2: Normalized (recall)
sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='RdYlGn', 
            vmin=0, vmax=100, cbar_kws={'label': 'Percentage (%)'},
            xticklabels=classes, yticklabels=classes, ax=ax2,
            linewidths=1, linecolor='white', square=True)
ax2.set_xlabel('Predicted Class', fontweight='bold')
ax2.set_ylabel('True Class', fontweight='bold')
ax2.set_title('(B) Normalized by True Class (Recall View)', fontweight='bold', pad=10)

# Add text annotations for key confusions
key_confusions = [
    (0, 1, '5.4%'),  # Glioma -> Meningioma
    (1, 0, '9.6%'),  # Meningioma -> Glioma
    (1, 2, '8.0%'),  # Meningioma -> No Tumor
    (1, 3, '8.8%'),  # Meningioma -> Pituitary
]

plt.tight_layout()
plt.savefig('Figure_2_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: Figure_2_Confusion_Matrix.png")

# ============================================================================
# FIGURE 3: Per-Class Performance with Clinical Context
# ============================================================================
print("\n[3/6] Generating Per-Class Performance Analysis...")

# Per-class metrics
metrics_data = {
    'Class': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
    'Precision': [80.81, 76.43, 83.13, 81.01],
    'Recall': [85.50, 73.60, 81.80, 80.60],
    'F1-Score': [83.09, 74.99, 82.46, 80.80],
    'Clinical Impact': ['High', 'Critical', 'Moderate', 'Moderate']
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Grouped bar chart
x = np.arange(len(metrics_data['Class']))
width = 0.25

bars1 = ax1.bar(x - width, metrics_data['Precision'], width, label='Precision',
                color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x, metrics_data['Recall'], width, label='Recall',
                color=COLORS['secondary'], alpha=0.8, edgecolor='black', linewidth=0.5)
bars3 = ax1.bar(x + width, metrics_data['F1-Score'], width, label='F1-Score',
                color=COLORS['success'], alpha=0.8, edgecolor='black', linewidth=0.5)

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

ax1.set_xlabel('Tumor Class', fontweight='bold')
ax1.set_ylabel('Performance (%)', fontweight='bold')
ax1.set_title('(A) Per-Class Metrics Comparison', fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_data['Class'])
ax1.legend(loc='lower left', framealpha=0.9)
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
ax1.set_ylim(0, 100)

# Add horizontal reference lines
ax1.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.5, label='80% Threshold')

# Subplot 2: Recall with error analysis
recall_values = metrics_data['Recall']
errors = [14.5, 26.4, 18.2, 19.4]  # Error rates (100 - recall)

colors_impact = {'High': '#d62728', 'Critical': '#ff9896', 'Moderate': '#9467bd'}
bar_colors = [colors_impact[impact] for impact in metrics_data['Clinical Impact']]

bars = ax2.barh(metrics_data['Class'], recall_values, color=bar_colors, 
                alpha=0.8, edgecolor='black', linewidth=1)

# Add error indicators
for i, (recall, error) in enumerate(zip(recall_values, errors)):
    ax2.barh(i, error, left=recall, color='lightgray', alpha=0.5, 
             edgecolor='black', linewidth=0.5)
    ax2.text(recall + error/2, i, f'{error:.1f}%\nerror', 
             ha='center', va='center', fontsize=8)
    ax2.text(recall/2, i, f'{recall:.1f}%', 
             ha='center', va='center', fontsize=9, fontweight='bold', color='white')

ax2.set_xlabel('Percentage (%)', fontweight='bold')
ax2.set_ylabel('Tumor Class', fontweight='bold')
ax2.set_title('(B) Recall with Error Rate Distribution', fontweight='bold', pad=10)
ax2.set_xlim(0, 100)
ax2.grid(True, alpha=0.3, axis='x', linestyle='--')

# Add legend for clinical impact
legend_elements = [
    mpatches.Patch(color=colors_impact['High'], label='High Impact (Malignant)'),
    mpatches.Patch(color=colors_impact['Critical'], label='Critical (Missed Diagnosis)'),
    mpatches.Patch(color=colors_impact['Moderate'], label='Moderate Impact')
]
ax2.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=8)

plt.tight_layout()
plt.savefig('Figure_3_PerClass_Performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: Figure_3_PerClass_Performance.png")

# ============================================================================
# FIGURE 4: Model Architecture Diagram
# ============================================================================
print("\n[4/6] Generating Model Architecture Diagram...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'Prototypical Networks Architecture', 
        ha='center', va='top', fontsize=16, fontweight='bold')

# Input
input_box = FancyBboxPatch((0.5, 7.5), 1.5, 1, boxstyle="round,pad=0.1", 
                           edgecolor=COLORS['primary'], facecolor='lightblue', linewidth=2)
ax.add_patch(input_box)
ax.text(1.25, 8, 'Input MRI\n128×128', ha='center', va='center', fontsize=9, fontweight='bold')

# Conv Block 1
conv1_box = FancyBboxPatch((2.5, 7.5), 1.3, 1, boxstyle="round,pad=0.1",
                           edgecolor=COLORS['success'], facecolor='lightgreen', linewidth=2)
ax.add_patch(conv1_box)
ax.text(3.15, 8.3, 'Conv Block 1', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(3.15, 7.9, '1→64 ch', ha='center', va='center', fontsize=7)
ax.text(3.15, 7.6, '64×64×64', ha='center', va='center', fontsize=7, style='italic')

# Conv Block 2
conv2_box = FancyBboxPatch((4.2, 7.5), 1.3, 1, boxstyle="round,pad=0.1",
                           edgecolor=COLORS['success'], facecolor='lightgreen', linewidth=2)
ax.add_patch(conv2_box)
ax.text(4.85, 8.3, 'Conv Block 2', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(4.85, 7.9, '64→128 ch', ha='center', va='center', fontsize=7)
ax.text(4.85, 7.6, '32×32×128', ha='center', va='center', fontsize=7, style='italic')

# Conv Block 3
conv3_box = FancyBboxPatch((5.9, 7.5), 1.3, 1, boxstyle="round,pad=0.1",
                           edgecolor=COLORS['success'], facecolor='lightgreen', linewidth=2)
ax.add_patch(conv3_box)
ax.text(6.55, 8.3, 'Conv Block 3', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(6.55, 7.9, '128→256 ch', ha='center', va='center', fontsize=7)
ax.text(6.55, 7.6, '16×16×256', ha='center', va='center', fontsize=7, style='italic')

# Conv Block 4
conv4_box = FancyBboxPatch((7.6, 7.5), 1.3, 1, boxstyle="round,pad=0.1",
                           edgecolor=COLORS['success'], facecolor='lightgreen', linewidth=2)
ax.add_patch(conv4_box)
ax.text(8.25, 8.3, 'Conv Block 4', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(8.25, 7.9, '256→512 ch', ha='center', va='center', fontsize=7)
ax.text(8.25, 7.6, '8×8×512', ha='center', va='center', fontsize=7, style='italic')

# Global Avg Pool
gap_box = FancyBboxPatch((9.3, 7.5), 1.2, 1, boxstyle="round,pad=0.1",
                         edgecolor=COLORS['warning'], facecolor='lightyellow', linewidth=2)
ax.add_patch(gap_box)
ax.text(9.9, 8.3, 'Global Avg', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(9.9, 7.9, 'Pooling', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(9.9, 7.6, '[512]', ha='center', va='center', fontsize=7, style='italic')

# FC Layers
fc_box = FancyBboxPatch((10.9, 7.5), 1.2, 1, boxstyle="round,pad=0.1",
                        edgecolor=COLORS['info'], facecolor='lavender', linewidth=2)
ax.add_patch(fc_box)
ax.text(11.5, 8.3, 'FC Layers', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(11.5, 7.9, '512→256→128', ha='center', va='center', fontsize=7)
ax.text(11.5, 7.6, 'L2 Norm', ha='center', va='center', fontsize=7, style='italic')

# Embedding
embed_box = FancyBboxPatch((12.5, 7.5), 1.2, 1, boxstyle="round,pad=0.1",
                           edgecolor=COLORS['danger'], facecolor='lightcoral', linewidth=2)
ax.add_patch(embed_box)
ax.text(13.1, 8.3, 'Embedding', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(13.1, 7.9, 'Vector', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(13.1, 7.6, '[128-D]', ha='center', va='center', fontsize=7, style='italic')

# Arrows connecting blocks
arrow_props = dict(arrowstyle='->', lw=2, color='black')
for start_x, end_x in [(2, 2.5), (3.8, 4.2), (5.5, 5.9), (7.2, 7.6), (8.9, 9.3), (10.5, 10.9), (12.1, 12.5)]:
    ax.annotate('', xy=(end_x, 8), xytext=(start_x, 8), arrowprops=arrow_props)

# Parameter count annotation
ax.text(7, 6.8, 'Total Parameters: 1,715,968', ha='center', va='center',
        fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Prototypical Classification (bottom part)
ax.text(7, 6, 'Prototypical Classification', ha='center', va='center',
        fontsize=14, fontweight='bold')

# Support Set
support_box = FancyBboxPatch((0.5, 3.5), 2.5, 2, boxstyle="round,pad=0.1",
                             edgecolor='blue', facecolor='lightblue', linewidth=2)
ax.add_patch(support_box)
ax.text(1.75, 5.2, 'Support Set', ha='center', va='top', fontsize=10, fontweight='bold')
ax.text(1.75, 4.7, 'K=5 per class', ha='center', va='center', fontsize=8)
ax.text(1.75, 4.3, '4 classes × 5', ha='center', va='center', fontsize=8)
ax.text(1.75, 3.9, '= 20 images', ha='center', va='center', fontsize=8)

# Prototypes
proto_box = FancyBboxPatch((3.5, 3.5), 2.5, 2, boxstyle="round,pad=0.1",
                           edgecolor='green', facecolor='lightgreen', linewidth=2)
ax.add_patch(proto_box)
ax.text(4.75, 5.2, 'Prototypes', ha='center', va='top', fontsize=10, fontweight='bold')
ax.text(4.75, 4.5, 'c₁, c₂, c₃, c₄', ha='center', va='center', fontsize=9)
ax.text(4.75, 4, 'Class centroids', ha='center', va='center', fontsize=8)

# Query
query_box = FancyBboxPatch((6.5, 3.5), 2.5, 2, boxstyle="round,pad=0.1",
                           edgecolor='orange', facecolor='lightyellow', linewidth=2)
ax.add_patch(query_box)
ax.text(7.75, 5.2, 'Query Sample', ha='center', va='top', fontsize=10, fontweight='bold')
ax.text(7.75, 4.5, 'New patient MRI', ha='center', va='center', fontsize=8)
ax.text(7.75, 4, '→ Embedding', ha='center', va='center', fontsize=8)

# Distance
dist_box = FancyBboxPatch((9.5, 3.5), 2.5, 2, boxstyle="round,pad=0.1",
                          edgecolor='purple', facecolor='lavender', linewidth=2)
ax.add_patch(dist_box)
ax.text(10.75, 5.2, 'Distance Calc', ha='center', va='top', fontsize=10, fontweight='bold')
ax.text(10.75, 4.5, '||q - cₖ||²', ha='center', va='center', fontsize=9)
ax.text(10.75, 4, 'Euclidean dist', ha='center', va='center', fontsize=8)

# Classification
class_box = FancyBboxPatch((12.5, 3.5), 1.3, 2, boxstyle="round,pad=0.1",
                           edgecolor='red', facecolor='lightcoral', linewidth=2)
ax.add_patch(class_box)
ax.text(13.15, 5.2, 'Predict', ha='center', va='top', fontsize=10, fontweight='bold')
ax.text(13.15, 4.5, 'argmin', ha='center', va='center', fontsize=9)
ax.text(13.15, 4, 'distance', ha='center', va='center', fontsize=8)

# Arrows for bottom flow
arrow_props_bottom = dict(arrowstyle='->', lw=2.5, color='darkblue')
for start_x, end_x in [(3, 3.5), (6, 6.5), (9, 9.5), (12, 12.5)]:
    ax.annotate('', xy=(end_x, 4.5), xytext=(start_x, 4.5), arrowprops=arrow_props_bottom)

# Add formula
ax.text(7, 2.5, r'Classification: $\hat{y} = \arg\min_k ||f_\phi(x_{query}) - c_k||^2$',
        ha='center', va='center', fontsize=11, style='italic',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5))

# Add legend
legend_elements = [
    mpatches.Patch(color='lightgreen', label='Convolutional Layers'),
    mpatches.Patch(color='lightyellow', label='Pooling/Aggregation'),
    mpatches.Patch(color='lavender', label='Fully Connected'),
    mpatches.Patch(color='lightcoral', label='Output/Classification')
]
ax.legend(handles=legend_elements, loc='lower left', framealpha=0.9, fontsize=8)

# Add note
ax.text(7, 0.8, 'Note: Each Conv Block contains Conv2D → BatchNorm → ReLU → MaxPool',
        ha='center', va='center', fontsize=8, style='italic', color='gray')

plt.tight_layout()
plt.savefig('Figure_4_Architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: Figure_4_Architecture.png")

# ============================================================================
# FIGURE 5: Episode Structure Visualization
# ============================================================================
print("\n[5/6] Generating Episode Structure Illustration...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(6, 9.5, '4-Way 5-Shot 10-Query Episode Structure', 
        ha='center', va='top', fontsize=16, fontweight='bold')

# Episode box
episode_border = FancyBboxPatch((0.3, 0.5), 11.4, 8, boxstyle="round,pad=0.2",
                                edgecolor='black', facecolor='white', linewidth=3, linestyle='--')
ax.add_patch(episode_border)
ax.text(6, 8.3, 'Single Episode (Task)', ha='center', va='center',
        fontsize=12, fontweight='bold', style='italic')

# Support Set
ax.text(2.5, 7.5, 'SUPPORT SET (Training for this task)', ha='center', va='center',
        fontsize=11, fontweight='bold', color='blue')
ax.text(2.5, 7.1, 'K=5 samples per class', ha='center', va='center',
        fontsize=9, style='italic')

# Define classes and colors
classes_viz = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
class_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

y_start = 6.5
for i, (cls, color) in enumerate(zip(classes_viz, class_colors)):
    y_pos = y_start - i * 1.2
    
    # Class label
    ax.text(0.6, y_pos, cls, ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Support samples (5 small boxes)
    for j in range(5):
        x_pos = 1.5 + j * 0.4
        sample_box = Rectangle((x_pos, y_pos-0.15), 0.3, 0.3, 
                               facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
        ax.add_patch(sample_box)
        ax.text(x_pos + 0.15, y_pos, f's{j+1}', ha='center', va='center', 
               fontsize=6, color='white', fontweight='bold')

# Arrow from support to prototype
arrow1 = FancyArrowPatch((3.5, 4.5), (4.3, 4.5),
                        arrowstyle='->', mutation_scale=30, linewidth=3, color='darkblue')
ax.add_patch(arrow1)
ax.text(3.9, 4.8, 'Compute\nPrototypes', ha='center', va='bottom', fontsize=8)

# Prototypes
ax.text(5.5, 7.5, 'PROTOTYPES', ha='center', va='center',
        fontsize=11, fontweight='bold', color='green')
ax.text(5.5, 7.1, 'Class centroids (mean embeddings)', ha='center', va='center',
        fontsize=8, style='italic')

for i, (cls, color) in enumerate(zip(classes_viz, class_colors)):
    y_pos = y_start - i * 1.2
    
    # Prototype circle
    proto_circle = Circle((5.5, y_pos), 0.25, facecolor=color, 
                         edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(proto_circle)
    ax.text(5.5, y_pos, f'c{i+1}', ha='center', va='center', 
           fontsize=9, color='white', fontweight='bold')

# Query Set
ax.text(9, 7.5, 'QUERY SET (Testing for this task)', ha='center', va='center',
        fontsize=11, fontweight='bold', color='orange')
ax.text(9, 7.1, 'N=10 samples per class', ha='center', va='center',
        fontsize=9, style='italic')

for i, (cls, color) in enumerate(zip(classes_viz, class_colors)):
    y_pos = y_start - i * 1.2
    
    # Query samples (10 small boxes)
    for j in range(10):
        x_pos = 7 + j * 0.4
        query_box = Rectangle((x_pos, y_pos-0.15), 0.3, 0.3,
                             facecolor=color, edgecolor='black', linewidth=1, alpha=0.5)
        ax.add_patch(query_box)
        ax.text(x_pos + 0.15, y_pos, f'q{j+1}', ha='center', va='center',
               fontsize=5, fontweight='bold')

# Arrow from prototype to query
arrow2 = FancyArrowPatch((6, 4.5), (6.8, 4.5),
                        arrowstyle='->', mutation_scale=30, linewidth=3, color='darkorange')
ax.add_patch(arrow2)
ax.text(6.4, 4.8, 'Classify\nQueries', ha='center', va='bottom', fontsize=8)

# Summary statistics
summary_box = FancyBboxPatch((3, 1.5), 6, 1.2, boxstyle="round,pad=0.15",
                             edgecolor='purple', facecolor='lavender', linewidth=2)
ax.add_patch(summary_box)
ax.text(6, 2.4, 'Episode Summary', ha='center', va='top', fontsize=10, fontweight='bold')
ax.text(6, 2.0, 'Support: 4 classes × 5 samples = 20 images', ha='center', va='center', fontsize=9)
ax.text(6, 1.7, 'Query: 4 classes × 10 samples = 40 images', ha='center', va='center', fontsize=9)
ax.text(6, 1.4, 'Total: 60 images per episode', ha='center', va='center', fontsize=9, fontweight='bold')

# Meta-learning note
ax.text(6, 0.6, 'Meta-Learning: Train on 1,000 diverse episodes → Learn to classify with minimal samples',
        ha='center', va='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('Figure_5_Episode_Structure.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: Figure_5_Episode_Structure.png")

# ============================================================================
# FIGURE 6: Clinical Decision Support Workflow
# ============================================================================
print("\n[6/6] Generating Clinical Workflow Diagram...")

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(6, 11.5, 'Few-Shot Meta-Learning Clinical Deployment Workflow', 
        ha='center', va='top', fontsize=16, fontweight='bold')

# Step 1: New Hospital
step1_box = FancyBboxPatch((1, 9.5), 2.5, 1.2, boxstyle="round,pad=0.15",
                           edgecolor='blue', facecolor='lightblue', linewidth=2)
ax.add_patch(step1_box)
ax.text(2.25, 10.5, '1. New Hospital', ha='center', va='top', fontsize=10, fontweight='bold')
ax.text(2.25, 10.1, 'Limited labeled', ha='center', va='center', fontsize=8)
ax.text(2.25, 9.8, 'MRI data', ha='center', va='center', fontsize=8)

# Arrow
arrow1 = FancyArrowPatch((3.5, 10.1), (5, 10.1),
                        arrowstyle='->', mutation_scale=30, linewidth=2.5, color='black')
ax.add_patch(arrow1)

# Step 2: Collect Support Set
step2_box = FancyBboxPatch((5, 9.5), 2.5, 1.2, boxstyle="round,pad=0.15",
                           edgecolor='green', facecolor='lightgreen', linewidth=2)
ax.add_patch(step2_box)
ax.text(6.25, 10.5, '2. Collect K=5', ha='center', va='top', fontsize=10, fontweight='bold')
ax.text(6.25, 10.1, 'Labeled samples', ha='center', va='center', fontsize=8)
ax.text(6.25, 9.8, 'per tumor class', ha='center', va='center', fontsize=8)

# Arrow
arrow2 = FancyArrowPatch((7.5, 10.1), (9, 10.1),
                        arrowstyle='->', mutation_scale=30, linewidth=2.5, color='black')
ax.add_patch(arrow2)

# Step 3: Deploy Model
step3_box = FancyBboxPatch((9, 9.5), 2.5, 1.2, boxstyle="round,pad=0.15",
                           edgecolor='orange', facecolor='lightyellow', linewidth=2)
ax.add_patch(step3_box)
ax.text(10.25, 10.5, '3. Deploy Model', ha='center', va='top', fontsize=10, fontweight='bold')
ax.text(10.25, 10.1, 'Load pre-trained', ha='center', va='center', fontsize=8)
ax.text(10.25, 9.8, 'embedding network', ha='center', va='center', fontsize=8)

# Vertical arrow
arrow3 = FancyArrowPatch((6, 9.3), (6, 8),
                        arrowstyle='->', mutation_scale=30, linewidth=2.5, color='black')
ax.add_patch(arrow3)

# Step 4: Compute Prototypes
step4_box = FancyBboxPatch((4.5, 6.5), 3, 1.5, boxstyle="round,pad=0.15",
                           edgecolor='purple', facecolor='lavender', linewidth=2)
ax.add_patch(step4_box)
ax.text(6, 7.7, '4. Compute Local Prototypes', ha='center', va='top', fontsize=10, fontweight='bold')
ax.text(6, 7.3, 'c_k = (1/5) Σ f_φ(x_i^k)', ha='center', va='center', fontsize=9, family='monospace')
ax.text(6, 6.9, 'From 5 local samples', ha='center', va='center', fontsize=8, style='italic')

# Vertical arrow
arrow4 = FancyArrowPatch((6, 6.3), (6, 5.2),
                        arrowstyle='->', mutation_scale=30, linewidth=2.5, color='black')
ax.add_patch(arrow4)

# Step 5: Inference
step5_box = FancyBboxPatch((3.5, 3.5), 5, 1.7, boxstyle="round,pad=0.15",
                           edgecolor='red', facecolor='lightcoral', linewidth=2)
ax.add_patch(step5_box)
ax.text(6, 5, '5. Inference on New Patients', ha='center', va='top', fontsize=10, fontweight='bold')
ax.text(6, 4.6, 'For each new MRI:', ha='center', va='center', fontsize=9)
ax.text(6, 4.3, '• Compute embedding: e = f_φ(x)', ha='left', va='center', fontsize=8)
ax.text(6, 4.0, '• Calculate distances: d_k = ||e - c_k||²', ha='left', va='center', fontsize=8)
ax.text(6, 3.7, '• Predict: ŷ = argmin_k(d_k)', ha='left', va='center', fontsize=8)

# Branching arrows
arrow5a = FancyArrowPatch((4, 3.3), (2.5, 2.2),
                         arrowstyle='->', mutation_scale=25, linewidth=2, color='darkgreen')
ax.add_patch(arrow5a)
ax.text(3, 2.8, 'High\nConfidence', ha='center', va='center', fontsize=7, 
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

arrow5b = FancyArrowPatch((8, 3.3), (9.5, 2.2),
                         arrowstyle='->', mutation_scale=25, linewidth=2, color='darkred')
ax.add_patch(arrow5b)
ax.text(9, 2.8, 'Low\nConfidence', ha='center', va='center', fontsize=7,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# Step 6a: Automated Report
step6a_box = FancyBboxPatch((0.5, 0.5), 3.5, 1.5, boxstyle="round,pad=0.15",
                            edgecolor='green', facecolor='lightgreen', linewidth=2)
ax.add_patch(step6a_box)
ax.text(2.25, 1.8, '6a. AI-Assisted Report', ha='center', va='top', fontsize=10, fontweight='bold')
ax.text(2.25, 1.4, '✓ Predicted class', ha='left', va='center', fontsize=8)
ax.text(2.25, 1.1, '✓ Confidence: 85%+', ha='left', va='center', fontsize=8)
ax.text(2.25, 0.8, '✓ Automated screening', ha='left', va='center', fontsize=8)

# Step 6b: Radiologist Review
step6b_box = FancyBboxPatch((8, 0.5), 3.5, 1.5, boxstyle="round,pad=0.15",
                            edgecolor='red', facecolor='mistyrose', linewidth=2)
ax.add_patch(step6b_box)
ax.text(9.75, 1.8, '6b. Radiologist Review', ha='center', va='top', fontsize=10, fontweight='bold')
ax.text(9.75, 1.4, '⚠ Ambiguous case', ha='left', va='center', fontsize=8)
ax.text(9.75, 1.1, '⚠ Confidence: <85%', ha='left', va='center', fontsize=8)
ax.text(9.75, 0.8, '⚠ Manual evaluation', ha='left', va='center', fontsize=8)

# Advantages box
adv_box = FancyBboxPatch((0.5, 8), 3, 1, boxstyle="round,pad=0.1",
                         edgecolor='darkblue', facecolor='aliceblue', linewidth=1.5)
ax.add_patch(adv_box)
ax.text(2, 8.7, 'Advantages:', ha='center', va='top', fontsize=9, fontweight='bold')
ax.text(2, 8.4, '✓ Only 5 samples needed', ha='left', va='center', fontsize=7)
ax.text(2, 8.15, '✓ Fast deployment (<1 day)', ha='left', va='center', fontsize=7)

# Performance box
perf_box = FancyBboxPatch((8.5, 8), 3, 1, boxstyle="round,pad=0.1",
                          edgecolor='darkgreen', facecolor='honeydew', linewidth=1.5)
ax.add_patch(perf_box)
ax.text(10, 8.7, 'Performance:', ha='center', va='top', fontsize=9, fontweight='bold')
ax.text(10, 8.4, '✓ 80.38% accuracy', ha='left', va='center', fontsize=7)
ax.text(10, 8.15, '✓ ~40ms inference time', ha='left', va='center', fontsize=7)

plt.tight_layout()
plt.savefig('Figure_6_Clinical_Workflow.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: Figure_6_Clinical_Workflow.png")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("FIGURE GENERATION COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  1. Figure_1_Training_Curves.png       - Training & validation learning curves")
print("  2. Figure_2_Confusion_Matrix.png      - Enhanced confusion matrix (absolute + normalized)")
print("  3. Figure_3_PerClass_Performance.png  - Per-class metrics with clinical context")
print("  4. Figure_4_Architecture.png          - Model architecture diagram")
print("  5. Figure_5_Episode_Structure.png     - Episode-based meta-learning illustration")
print("  6. Figure_6_Clinical_Workflow.png     - Clinical deployment workflow")
print("\nAll figures saved at 300 DPI (publication quality)")
print("Ready for SINTA 1 journal submission!")
print("=" * 80)
