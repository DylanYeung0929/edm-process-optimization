"""
BIT Mathematical Modeling Competition 2026 - Problem B
EDM Process Optimization: Complete Figure Generation Script (ver2)
新增：论文所需全部关键数值计算与输出（不改动图表生成逻辑）
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = '../202601B题数据.xlsx'
OUT = 'outputs/'

# ── Font & Style Setup ──────────────────────────────────────────────────────
FONT = 'DejaVu Serif'   # closest to Times New Roman available on this system
plt.rcParams.update({
    'font.family': FONT,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

PALETTE = {
    'blue':   '#2C6FAC',
    'red':    '#D94F3D',
    'green':  '#3A9E6C',
    'orange': '#E87C2A',
    'purple': '#7B5EA7',
    'gray':   '#7F8C8D',
    'teal':   '#16A085',
    'gold':   '#D4AC0D',
}
CLASS_COLORS = {-1: PALETTE['red'], 0: PALETTE['gray'], 1: PALETTE['green']}


# ── Load Data ───────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_FILE)
features = [c for c in df.columns if c not in ['DFlow', 'DGap']]
X = df[features].values
y_flow = df['DFlow'].values
y_gap  = df['DGap'].values
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

print("Data loaded:", df.shape)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Data Overview: Distribution + Class Balance
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.55, wspace=0.45)

# (a) Class distribution pie charts
ax_pie1 = fig.add_subplot(gs[0, 0:2])
ax_pie2 = fig.add_subplot(gs[0, 2:4])

vals_flow  = pd.Series(y_flow).value_counts().sort_index()
vals_gap   = pd.Series(y_gap).value_counts().sort_index()
colors_pie = [CLASS_COLORS[k] for k in vals_flow.index]

wedges1, texts1, autotexts1 = ax_pie1.pie(
    vals_flow.values, labels=[f'DFlow={k}' for k in vals_flow.index],
    colors=colors_pie, autopct='%1.1f%%', startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
for at in autotexts1: at.set_fontsize(8.5)
ax_pie1.set_title('(a) DFlow Class Distribution', fontweight='bold')

colors_pie2 = [CLASS_COLORS[k] for k in vals_gap.index]
wedges2, texts2, autotexts2 = ax_pie2.pie(
    vals_gap.values, labels=[f'DGap={k}' for k in vals_gap.index],
    colors=colors_pie2, autopct='%1.1f%%', startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
for at in autotexts2: at.set_fontsize(8.5)
ax_pie2.set_title('(b) DGap Class Distribution', fontweight='bold')

# (c) Joint state heatmap
ax_heat = fig.add_subplot(gs[0, 4:6])
combo = df.groupby(['DFlow', 'DGap']).size().unstack(fill_value=0)
im = ax_heat.imshow(combo.values, cmap='Blues', aspect='auto')
ax_heat.set_xticks(range(len(combo.columns))); ax_heat.set_xticklabels(combo.columns)
ax_heat.set_yticks(range(len(combo.index)));   ax_heat.set_yticklabels(combo.index)
ax_heat.set_xlabel('DGap'); ax_heat.set_ylabel('DFlow')
ax_heat.set_title('(c) Joint State Frequency', fontweight='bold')
for i in range(combo.shape[0]):
    for j in range(combo.shape[1]):
        ax_heat.text(j, i, str(combo.values[i, j]), ha='center', va='center', fontsize=11,
                     color='white' if combo.values[i, j] > 30 else 'black', fontweight='bold')
plt.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.04)

# (d–s) Feature distributions (violin by DGap class)
for idx, feat in enumerate(features):
    row = 1 + idx // 6
    col = idx % 6
    if row > 2: break
    ax = fig.add_subplot(gs[row, col])
    groups = [df[df['DGap'] == c][feat].values for c in [-1, 0, 1]]
    parts = ax.violinplot(groups, positions=[0, 1, 2], showmedians=True, widths=0.7)
    for i, (pc, c) in enumerate(zip(parts['bodies'], [-1, 0, 1])):
        pc.set_facecolor(CLASS_COLORS[c]); pc.set_alpha(0.65)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.5)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['-1', '0', '1'], fontsize=7)
    ax.set_title(feat, fontsize=7.5, fontweight='bold')
    ax.set_xlabel('DGap', fontsize=7)
    ax.tick_params(labelsize=7)

patches = [mpatches.Patch(color=CLASS_COLORS[c], label=f'DGap={c}') for c in [-1, 0, 1]]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=9,
           title='DGap Class', title_fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

plt.suptitle('Figure 2: Dataset Overview — Class Distributions and Feature Distributions by DGap',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig(OUT + 'Fig1_Data_Overview.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig1 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Correlation Analysis
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: feature-feature correlation heatmap
corr = df[features].corr()
cmap = LinearSegmentedColormap.from_list('rdb', ['#D94F3D', 'white', '#2C6FAC'])
im = axes[0].imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
axes[0].set_xticks(range(16)); axes[0].set_xticklabels(features, rotation=45, ha='right', fontsize=7.5)
axes[0].set_yticks(range(16)); axes[0].set_yticklabels(features, fontsize=7.5)
axes[0].set_title('(a) Feature Inter-Correlation Matrix', fontweight='bold')
for i in range(16):
    for j in range(16):
        v = corr.values[i, j]
        axes[0].text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5.5,
                     color='white' if abs(v) > 0.65 else 'black')
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='Pearson r')

# Right: grouped bar chart
corr_flow = df[features].corrwith(df['DFlow'])
corr_gap  = df[features].corrwith(df['DGap'])
x = np.arange(16); w = 0.38
b1 = axes[1].barh(x - w/2, corr_flow.values, w, color=PALETTE['blue'],  alpha=0.85, label='DFlow')
b2 = axes[1].barh(x + w/2, corr_gap.values,  w, color=PALETTE['red'],   alpha=0.85, label='DGap')
axes[1].set_yticks(x); axes[1].set_yticklabels(features, fontsize=8.5)
axes[1].axvline(0, color='black', lw=0.8, ls='--')
axes[1].set_xlabel('Pearson Correlation Coefficient', fontsize=11)
axes[1].set_title('(b) Feature Correlations with DFlow & DGap', fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].set_xlim(-0.85, 0.85)
axes[1].grid(axis='x', alpha=0.3, ls='--')
for bar in list(b1) + list(b2):
    w_val = bar.get_width()
    axes[1].text(w_val + 0.01*(1 if w_val >= 0 else -1),
                 bar.get_y() + bar.get_height()/2,
                 f'{w_val:.2f}', va='center',
                 ha='left' if w_val >= 0 else 'right', fontsize=6)

plt.suptitle('Figure 3: Pearson Correlation Analysis of Pulse Features and Control Variables',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig2_Correlation_Analysis.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig2 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 – PCA Visualization (2D + 3D)
# ═══════════════════════════════════════════════════════════════════════════
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_sc)

fig = plt.figure(figsize=(18, 6))

# (a) Explained variance scree plot
ax1 = fig.add_subplot(131)
var_exp = pca.explained_variance_ratio_ * 100
cum_var = np.cumsum(var_exp)
ax1.bar(range(1, 7), var_exp, color=PALETTE['blue'], alpha=0.8, label='Individual')
ax1.plot(range(1, 7), cum_var, 'o-', color=PALETTE['red'], lw=2, label='Cumulative')
ax1.axhline(80, color=PALETTE['orange'], ls='--', lw=1.2, label='80% threshold')
ax1.set_xlabel('Principal Component', fontsize=11)
ax1.set_ylabel('Explained Variance (%)', fontsize=11)
ax1.set_title('(a) PCA Scree Plot', fontweight='bold')
ax1.legend()
ax1.set_xticks(range(1, 7))
for i, (v, c) in enumerate(zip(var_exp, cum_var)):
    ax1.text(i+1, v+0.5, f'{v:.1f}%', ha='center', fontsize=8)

# (b) 2D PCA scatter colored by DGap
ax2 = fig.add_subplot(132)
for cls in [-1, 0, 1]:
    mask = y_gap == cls
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=CLASS_COLORS[cls], label=f'DGap={cls}',
                alpha=0.75, s=45, edgecolors='white', linewidth=0.4)
ax2.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)', fontsize=11)
ax2.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)', fontsize=11)
ax2.set_title('(b) PCA Score Plot — DGap Classes', fontweight='bold')
ax2.legend()

# (c) 3D PCA scatter colored by DGap
ax3 = fig.add_subplot(133, projection='3d')
for cls in [-1, 0, 1]:
    mask = y_gap == cls
    ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                c=CLASS_COLORS[cls], label=f'DGap={cls}',
                alpha=0.75, s=35, edgecolors='none')
ax3.set_xlabel(f'PC1\n({var_exp[0]:.1f}%)', fontsize=8, labelpad=8)
ax3.set_ylabel(f'PC2\n({var_exp[1]:.1f}%)', fontsize=8, labelpad=8)
ax3.set_zlabel(f'PC3\n({var_exp[2]:.1f}%)', fontsize=8, labelpad=8)
ax3.set_title('(c) 3D PCA Score Plot — DGap Classes', fontweight='bold')
ax3.legend(fontsize=8)
ax3.view_init(elev=22, azim=135)

plt.suptitle('Figure 6: Principal Component Analysis of Pulse Feature Space',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig3_PCA_Analysis.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig3 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Random Forest Feature Importance
# ═══════════════════════════════════════════════════════════════════════════
rf_flow = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_gap  = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_flow.fit(X_sc, y_flow)
rf_gap.fit(X_sc, y_gap)

imp_flow = rf_flow.feature_importances_
imp_gap  = rf_gap.feature_importances_

# Sort by DGap importance
order = np.argsort(imp_gap)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

colors_flow = [PALETTE['blue'] if v > np.median(imp_flow) else '#A8C8E8' for v in imp_flow[order]]
colors_gap  = [PALETTE['red']  if v > np.median(imp_gap)  else '#F4A7A0' for v in imp_gap[order]]

b1 = axes[0].barh(range(16), imp_flow[order]*100, color=colors_flow, edgecolor='white')
axes[0].set_yticks(range(16)); axes[0].set_yticklabels([features[i] for i in order], fontsize=9)
axes[0].set_xlabel('Feature Importance (%)', fontsize=11)
axes[0].set_title('(a) Random Forest Importance — DFlow', fontweight='bold')
axes[0].axvline(np.median(imp_flow)*100, ls='--', color='gray', lw=1, label='Median')
axes[0].legend()
for bar, val in zip(b1, imp_flow[order]*100):
    axes[0].text(val+0.1, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=7.5)

b2 = axes[1].barh(range(16), imp_gap[order]*100, color=colors_gap, edgecolor='white')
axes[1].set_yticks(range(16)); axes[1].set_yticklabels([features[i] for i in order], fontsize=9)
axes[1].set_xlabel('Feature Importance (%)', fontsize=11)
axes[1].set_title('(b) Random Forest Importance — DGap', fontweight='bold')
axes[1].axvline(np.median(imp_gap)*100, ls='--', color='gray', lw=1, label='Median')
axes[1].legend()
for bar, val in zip(b2, imp_gap[order]*100):
    axes[1].text(val+0.1, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=7.5)

plt.suptitle('Figure 5: Random Forest Feature Importance for DFlow and DGap Prediction',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig4_Feature_Importance.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig4 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 – 3D Feature Space Visualization (top 3 DGap features)
# ═══════════════════════════════════════════════════════════════════════════
top3_gap = [features[i] for i in np.argsort(imp_gap)[::-1][:3]]

fig = plt.figure(figsize=(16, 6))

for subplot_idx, (azim, elev, label) in enumerate([(135, 25, '(a)'), (45, 25, '(b)'), (90, 60, '(c)')]):
    ax = fig.add_subplot(1, 3, subplot_idx+1, projection='3d')
    for cls in [-1, 0, 1]:
        mask = y_gap == cls
        ax.scatter(df.loc[mask, top3_gap[0]], df.loc[mask, top3_gap[1]], df.loc[mask, top3_gap[2]],
                   c=CLASS_COLORS[cls], label=f'DGap={cls}', alpha=0.7, s=30, edgecolors='none')
    ax.set_xlabel(top3_gap[0], fontsize=7.5, labelpad=6)
    ax.set_ylabel(top3_gap[1], fontsize=7.5, labelpad=6)
    ax.set_zlabel(top3_gap[2], fontsize=7.5, labelpad=6)
    ax.set_title(f'{label} View {subplot_idx+1}', fontweight='bold')
    ax.view_init(elev=elev, azim=azim)
    if subplot_idx == 0:
        ax.legend(fontsize=8, loc='upper left')

plt.suptitle(f'Figure 14: 3D Scatter Plot of Top-3 DGap-Important Features ({", ".join(top3_gap)})',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig5_3D_Feature_Space.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig5 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6 – LDA Discriminant Analysis
# ═══════════════════════════════════════════════════════════════════════════
lda_gap  = LinearDiscriminantAnalysis()
lda_flow = LinearDiscriminantAnalysis()
X_lda_gap  = lda_gap.fit_transform(X_sc, y_gap)
X_lda_flow = lda_flow.fit_transform(X_sc, y_flow)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for cls in [-1, 0, 1]:
    mask = y_gap == cls
    if X_lda_gap.ndim == 1 or X_lda_gap.shape[1] == 1:
        axes[0].hist(X_lda_gap[mask].ravel(), bins=20, color=CLASS_COLORS[cls],
                     alpha=0.6, label=f'DGap={cls}', density=True)
    else:
        axes[0].scatter(X_lda_gap[mask, 0], X_lda_gap[mask, 1],
                        c=CLASS_COLORS[cls], label=f'DGap={cls}', alpha=0.7, s=45)
axes[0].set_title('(a) LDA Projection — DGap Classes', fontweight='bold')
axes[0].set_xlabel('LD1', fontsize=11); axes[0].set_ylabel('LD2' if X_lda_gap.shape[1] > 1 else 'Density', fontsize=11)
axes[0].legend()

for cls in [-1, 0, 1]:
    mask = y_flow == cls
    if X_lda_flow.ndim == 1 or X_lda_flow.shape[1] == 1:
        axes[1].hist(X_lda_flow[mask].ravel(), bins=15, color=CLASS_COLORS[cls],
                     alpha=0.6, label=f'DFlow={cls}', density=True)
    else:
        axes[1].scatter(X_lda_flow[mask, 0], X_lda_flow[mask, 1],
                        c=CLASS_COLORS[cls], label=f'DFlow={cls}', alpha=0.7, s=45)
axes[1].set_title('(b) LDA Projection — DFlow Classes', fontweight='bold')
axes[1].set_xlabel('LD1', fontsize=11); axes[1].set_ylabel('LD2' if X_lda_flow.shape[1] > 1 else 'Density', fontsize=11)
axes[1].legend()

plt.suptitle('Figure 7: Linear Discriminant Analysis — Class Separability in Reduced Space',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig6_LDA_Analysis.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig6 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7 – Model Performance: Cross-Validation
# ═══════════════════════════════════════════════════════════════════════════
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42),
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'LDA': LinearDiscriminantAnalysis(),
    'Logistic Reg.': LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results_flow, results_gap = {}, {}
for name, model in models.items():
    results_flow[name] = cross_val_score(model, X_sc, y_flow, cv=cv, scoring='f1_macro')
    results_gap[name]  = cross_val_score(model, X_sc, y_gap,  cv=cv, scoring='f1_macro')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, results, title in [(axes[0], results_flow, '(a) DFlow Prediction'),
                            (axes[1], results_gap,  '(b) DGap Prediction')]:
    names = list(results.keys())
    means = [np.mean(v)*100 for v in results.values()]
    stds  = [np.std(v)*100  for v in results.values()]
    colors_bar = [PALETTE['blue'], PALETTE['red'], PALETTE['green'], PALETTE['orange'], PALETTE['purple']]
    bars = ax.bar(names, means, yerr=stds, capsize=5,
                  color=colors_bar, alpha=0.85, edgecolor='white', error_kw={'elinewidth': 2})
    ax.set_ylim(0, 100)
    ax.set_ylabel('Macro F1-Score (%)', fontsize=11)
    ax.set_title(title, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.3, ls='--')
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x()+bar.get_width()/2, m+s+1.5, f'{m:.1f}±{s:.1f}',
                ha='center', fontsize=8, fontweight='bold')
    ax.axhline(max(means), ls=':', color='gold', lw=1.5, label=f'Best: {max(means):.1f}%')
    ax.legend(fontsize=9)

plt.suptitle('Figure 10: 5-Fold Cross-Validation Performance of Classification Models',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig7_Model_Performance.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig7 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8 – Confusion Matrices (Best model: Random Forest)
# ═══════════════════════════════════════════════════════════════════════════
from sklearn.model_selection import train_test_split

X_tr, X_te, yf_tr, yf_te, yg_tr, yg_te = train_test_split(
    X_sc, y_flow, y_gap, test_size=0.25, random_state=42,
    stratify=y_flow  # stratify by flow as it's more imbalanced
)

rf_f = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
rf_g = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
rf_f.fit(X_tr, yf_tr); rf_g.fit(X_tr, yg_tr)
yf_pred = rf_f.predict(X_te); yg_pred = rf_g.predict(X_te)

cm_flow = confusion_matrix(yf_te, yf_pred, labels=[-1, 0, 1])
cm_gap  = confusion_matrix(yg_te, yg_pred, labels=[-1, 0, 1])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cmap_cm = LinearSegmentedColormap.from_list('cm', ['white', PALETTE['blue']])

for ax, cm, title, ylabel in [
    (axes[0], cm_flow, '(a) DFlow — Confusion Matrix', 'True Label'),
    (axes[1], cm_gap,  '(b) DGap — Confusion Matrix',  'True Label')]:
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap_cm)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['-1', '0', '1'])
    ax.set_yticks([0,1,2]); ax.set_yticklabels(['-1', '0', '1'])
    ax.set_xlabel('Predicted Label', fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight='bold')
    thresh = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Figure 11: Confusion Matrices of Random Forest Classifier (Test Set)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig8_Confusion_Matrices.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig8 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 9 – Synergistic Control Analysis: Joint State Space
# ═══════════════════════════════════════════════════════════════════════════
# Define composite control index S = DFlow + DGap (ranges -2 to 2)
df['S'] = df['DFlow'] + df['DGap']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) scatter DFlow vs DGap with jitter
jitter = np.random.RandomState(0).normal(0, 0.06, len(df))
sc = axes[0].scatter(df['DFlow'] + jitter, df['DGap'] + jitter,
                     c=df['S'], cmap='RdBu_r', vmin=-2, vmax=2, alpha=0.7, s=50, edgecolors='white', lw=0.3)
axes[0].set_xlabel('DFlow', fontsize=11); axes[0].set_ylabel('DGap', fontsize=11)
axes[0].set_title('(a) DFlow vs DGap Joint State\n(Color = S = DFlow + DGap)', fontweight='bold')
axes[0].set_xticks([-1, 0, 1]); axes[0].set_yticks([-1, 0, 1])
plt.colorbar(sc, ax=axes[0], label='S = DFlow + DGap')

# (b) Box plots of top feature vs composite S
s_vals = sorted(df['S'].unique())
feat_plot = top3_gap[0]
data_s = [df[df['S'] == s][feat_plot].values for s in s_vals]
bp = axes[1].boxplot(data_s, labels=[str(s) for s in s_vals],
                     patch_artist=True, notch=False,
                     boxprops={'linewidth': 1.2},
                     whiskerprops={'linewidth': 1.2},
                     medianprops={'color': 'black', 'linewidth': 2})
colors_s = [PALETTE['red'], PALETTE['orange'], PALETTE['gray'], PALETTE['teal'], PALETTE['blue']]
for patch, col in zip(bp['boxes'], colors_s):
    patch.set_facecolor(col); patch.set_alpha(0.7)
axes[1].set_xlabel('Composite Control Index S = DFlow + DGap', fontsize=10)
axes[1].set_ylabel(feat_plot, fontsize=10)
axes[1].set_title(f'(b) {feat_plot} Distribution\nby Composite Index S', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3, ls='--')

# (c) Kruskal-Wallis test H-statistic for each feature vs S
h_stats, p_vals = [], []
for f in features:
    groups = [df[df['S'] == s][f].values for s in s_vals]
    h, p = stats.kruskal(*[g for g in groups if len(g) > 0])
    h_stats.append(h); p_vals.append(p)

colors_h = [PALETTE['red'] if p < 0.05 else PALETTE['gray'] for p in p_vals]
bars = axes[2].barh(range(16), h_stats, color=colors_h, alpha=0.85)
axes[2].set_yticks(range(16)); axes[2].set_yticklabels(features, fontsize=8.5)
axes[2].set_xlabel('Kruskal-Wallis H-Statistic', fontsize=11)
axes[2].set_title('(c) Feature Sensitivity to\nComposite Control Index S', fontweight='bold')
axes[2].axvline(stats.chi2.ppf(0.95, df=4), ls='--', color='black', lw=1.2, label='p=0.05 threshold')
axes[2].legend(fontsize=8)
sig_patch = mpatches.Patch(color=PALETTE['red'],  label='Significant (p<0.05)')
ns_patch  = mpatches.Patch(color=PALETTE['gray'], label='Not significant')
axes[2].legend(handles=[sig_patch, ns_patch], fontsize=8, loc='lower right')

plt.suptitle('Figure 8: Synergistic Control Analysis — Joint DFlow & DGap State Space',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig9_Synergy_Analysis.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig9 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 10 – Predictive Model Decision Boundary (top 2 features)
# ═══════════════════════════════════════════════════════════════════════════
top2_gap = [features[i] for i in np.argsort(imp_gap)[::-1][:2]]
X2 = df[top2_gap].values
sc2 = StandardScaler(); X2_sc = sc2.fit_transform(X2)

rf_2d = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_2d.fit(X2_sc, y_gap)

x_min, x_max = X2_sc[:, 0].min() - 0.5, X2_sc[:, 0].max() + 0.5
y_min, y_max = X2_sc[:, 1].min() - 0.5, X2_sc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = rf_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Decision boundary
cmap_bg = LinearSegmentedColormap.from_list('bg', ['#F4A7A0', '#D3D3D3', '#A8C8E8'])
axes[0].contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg, levels=[-1.5, -0.5, 0.5, 1.5])
axes[0].contour(xx, yy, Z, colors='gray', linewidths=0.8, levels=[-0.5, 0.5], linestyles='--')
for cls in [-1, 0, 1]:
    mask = y_gap == cls
    axes[0].scatter(X2_sc[mask, 0], X2_sc[mask, 1],
                    c=CLASS_COLORS[cls], label=f'DGap={cls}',
                    s=50, edgecolors='white', lw=0.5, alpha=0.85)
axes[0].set_xlabel(f'{top2_gap[0]} (standardized)', fontsize=10)
axes[0].set_ylabel(f'{top2_gap[1]} (standardized)', fontsize=10)
axes[0].set_title('(a) Random Forest Decision Boundary\n(Top-2 DGap Features)', fontweight='bold')
axes[0].legend()

# Probability surface for class DGap=1
proba = rf_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])
class_idx = list(rf_2d.classes_).index(1)
Z_proba = proba[:, class_idx].reshape(xx.shape)
contf = axes[1].contourf(xx, yy, Z_proba, levels=20, cmap='Blues', alpha=0.9)
plt.colorbar(contf, ax=axes[1], label='P(DGap = +1)')
for cls in [-1, 0, 1]:
    mask = y_gap == cls
    axes[1].scatter(X2_sc[mask, 0], X2_sc[mask, 1],
                    c=CLASS_COLORS[cls], label=f'DGap={cls}',
                    s=40, edgecolors='white', lw=0.4, alpha=0.8, marker=['v', 'o', '^'][cls+1])
axes[1].set_xlabel(f'{top2_gap[0]} (standardized)', fontsize=10)
axes[1].set_ylabel(f'{top2_gap[1]} (standardized)', fontsize=10)
axes[1].set_title('(b) Predicted Probability Surface\n[P(DGap = +1)]', fontweight='bold')
axes[1].legend()

plt.suptitle('Figure 12: Random Forest Decision Boundary and Probability Map for DGap Prediction',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig10_Decision_Boundary.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig10 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 11 – Hierarchical Clustering Dendrogram
# ═══════════════════════════════════════════════════════════════════════════
# Cluster features by their correlation pattern
corr_feat = df[features].corr().values
dist_feat = 1 - np.abs(corr_feat)
np.fill_diagonal(dist_feat, 0)

Z_link = linkage(dist_feat, method='average')

fig, ax = plt.subplots(figsize=(14, 6))
dend = dendrogram(Z_link, labels=features, ax=ax,
                  color_threshold=0.4,
                  leaf_rotation=45, leaf_font_size=10,
                  above_threshold_color=PALETTE['gray'])
ax.set_ylabel('Distance (1 - |Pearson r|)', fontsize=11)
ax.set_title('(a) Hierarchical Clustering of Pulse Features by Correlation Structure', fontweight='bold')
ax.axhline(0.4, ls='--', color='red', lw=1.2, label='Cluster cutoff')
ax.legend(fontsize=9)

# Add cluster labels
ax2 = ax.twinx()
ax2.set_visible(False)

plt.suptitle('Figure 4: Hierarchical Cluster Dendrogram — Feature Grouping by Correlation',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig11_Dendrogram.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig11 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 12 – Modeling Flowchart (SVG-style using matplotlib)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis('off')

def box(ax, x, y, w, h, text, color='#2C6FAC', text_color='white', fontsize=9.5, alpha=0.9, radius=0.25):
    fancy = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                    boxstyle=f"round,pad={radius}",
                                    facecolor=color, edgecolor='white', linewidth=1.5, alpha=alpha)
    ax.add_patch(fancy)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', wrap=True,
            multialignment='center')

def arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.8))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.1, my, label, fontsize=8, color='#555555', style='italic')

# Main flow
boxes = [
    (7, 8.3, 5.5, 0.7, 'Raw EDM Process Data\n(154 samples × 18 variables)', '#1A5276'),
    (7, 7.2, 5.5, 0.7, 'Data Preprocessing\n(Standardization, Outlier Check)', '#2471A3'),
    (7, 6.1, 5.5, 0.7, 'Exploratory Data Analysis\n(Correlation, Distribution, PCA)', '#2980B9'),
    (3, 4.6, 4.2, 0.85, 'Task 1: Feature–Control\nQuantitative Relationship\n(Pearson r, RF Importance)', '#1E8449'),
    (7, 4.6, 2.8, 0.85, 'Task 2: Synergy\nAnalysis\n(Joint State, KW Test)', '#B7950B'),
    (11, 4.6, 3.8, 0.85, 'Task 3: Optimal\nPrediction Model\n(RF Classifier)', '#922B21'),
    (3, 2.9, 4.2, 0.85, 'Feature Ranking &\nImportance Scores\n(16 features → DFlow, DGap)', '#1E8449'),
    (7, 2.9, 2.8, 0.85, 'Composite Control\nIndex S = DFlow+DGap\n(Simplified Workflow)', '#B7950B'),
    (11, 2.9, 3.8, 0.85, '5-Fold CV Validation\nF1-Score Evaluation\nDecision Boundary Map', '#922B21'),
    (7, 1.3, 8.5, 0.75, 'Conclusions: Quantitative Models for EDM Parameter Optimization', '#1A5276'),
]
for bx, by, bw, bh, bt, bc in boxes:
    box(ax, bx, by, bw, bh, bt, color=bc)

# Arrows
arrow(ax, 7, 7.95, 7, 7.55)
arrow(ax, 7, 6.85, 7, 6.45)
arrow(ax, 5, 5.85, 3.5, 5.03)
arrow(ax, 7, 5.85, 7, 5.03)
arrow(ax, 9, 5.85, 10.5, 5.03)
arrow(ax, 3, 4.18, 3, 3.33)
arrow(ax, 7, 4.18, 7, 3.33)
arrow(ax, 11, 4.18, 11, 3.33)
arrow(ax, 3, 2.48, 5.5, 1.68)
arrow(ax, 7, 2.48, 7, 1.68)
arrow(ax, 11, 2.48, 8.5, 1.68)

ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('#F8F9FA')
plt.title('Figure 1: Mathematical Modeling Framework for EDM Process Optimization',
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(OUT + 'Fig12_Flowchart.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig12 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 13 – Box-plots of Feature Values across DFlow states (extended)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes = axes.ravel()

for i, feat in enumerate(features):
    ax = axes[i]
    groups = [df[df['DFlow'] == c][feat].values for c in [-1, 0, 1]]
    bp = ax.boxplot(groups, labels=['-1', '0', '1'],
                    patch_artist=True, notch=False,
                    boxprops={'linewidth': 1.2},
                    medianprops={'color': 'black', 'linewidth': 2},
                    whiskerprops={'linewidth': 1})
    clrs = [CLASS_COLORS[-1], CLASS_COLORS[0], CLASS_COLORS[1]]
    for patch, col in zip(bp['boxes'], clrs):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.set_title(feat, fontsize=9, fontweight='bold')
    ax.set_xlabel('DFlow State', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', alpha=0.3, ls='--')
    # Kruskal-Wallis test
    h, p = stats.kruskal(*[g for g in groups if len(g) > 0])
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    ax.set_xlabel(f'DFlow State  (KW: {sig})', fontsize=7.5)

patches = [mpatches.Patch(color=CLASS_COLORS[c], label=f'DFlow={c}') for c in [-1, 0, 1]]
fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10, title='DFlow Class',
           title_fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))
plt.suptitle('Figure 9: Box-Plot Analysis of All 16 Pulse Features Stratified by DFlow State',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + 'Fig13_Boxplots_DFlow.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig13 saved")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 14 – 3D Surface: Probability of optimal state vs top 2 PCA components
# ═══════════════════════════════════════════════════════════════════════════
pca2 = PCA(n_components=2); X_pca2 = pca2.fit_transform(X_sc)
rf_pca = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_pca.fit(X_pca2, y_gap)

x1_range = np.linspace(X_pca2[:, 0].min()-0.5, X_pca2[:, 0].max()+0.5, 80)
x2_range = np.linspace(X_pca2[:, 1].min()-0.5, X_pca2[:, 1].max()+0.5, 80)
xx1, xx2 = np.meshgrid(x1_range, x2_range)
grid_pts  = np.c_[xx1.ravel(), xx2.ravel()]
proba_surf = rf_pca.predict_proba(grid_pts)
opt_idx    = list(rf_pca.classes_).index(1)
Z_surf     = proba_surf[:, opt_idx].reshape(xx1.shape)

fig = plt.figure(figsize=(16, 6))

ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(xx1, xx2, Z_surf, cmap='coolwarm', alpha=0.85, edgecolor='none')
ax1.scatter(X_pca2[:, 0], X_pca2[:, 1], [0]*len(X_pca2),
            c=[CLASS_COLORS[c] for c in y_gap], s=20, alpha=0.5, zorder=5)
ax1.set_xlabel('PC1', fontsize=9, labelpad=8)
ax1.set_ylabel('PC2', fontsize=9, labelpad=8)
ax1.set_zlabel('P(DGap=+1)', fontsize=9, labelpad=8)
ax1.set_title('(a) 3D Probability Surface\nP(DGap=+1) in PCA Space', fontweight='bold')
fig.colorbar(surf, ax=ax1, fraction=0.03, pad=0.08, label='Probability')
ax1.view_init(elev=30, azim=220)

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(xx1, xx2, Z_surf, cmap='viridis', alpha=0.85, edgecolor='none')
for cls in [-1, 0, 1]:
    mask = y_gap == cls
    z_cls = rf_pca.predict_proba(X_pca2[mask])[:, opt_idx]
    ax2.scatter(X_pca2[mask, 0], X_pca2[mask, 1], z_cls,
                c=CLASS_COLORS[cls], s=30, label=f'DGap={cls}', alpha=0.9, zorder=10)
ax2.set_xlabel('PC1', fontsize=9, labelpad=8)
ax2.set_ylabel('PC2', fontsize=9, labelpad=8)
ax2.set_zlabel('P(DGap=+1)', fontsize=9, labelpad=8)
ax2.set_title('(b) Actual Data Points on\nProbability Surface', fontweight='bold')
ax2.legend(fontsize=8, loc='upper left')
ax2.view_init(elev=25, azim=60)

plt.suptitle('Figure 13: 3D Probability Surface for Optimal DGap State Prediction in PCA Feature Space',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig14_3D_Probability_Surface.png', dpi=160, bbox_inches='tight')
plt.close()
print("Fig14 saved")

print("\n✓ All 14 figures generated successfully!")

# ═══════════════════════════════════════════════════════════════════════════
# ★ 数值计算模块（ver2 新增）——论文所需全部关键数值
# ═══════════════════════════════════════════════════════════════════════════
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json

SEP = "=" * 70

print(f"\n{SEP}")
print("★  数值计算模块  ★")
print(SEP)

# ── 1. 数据集基本统计 ────────────────────────────────────────────────────
print("\n【1】数据集基本统计")
print(f"  样本总数: {len(df)}")
vals_flow_cnt = pd.Series(y_flow).value_counts().sort_index()
vals_gap_cnt  = pd.Series(y_gap).value_counts().sort_index()
print(f"  DFlow 分布: { {int(k): int(v) for k, v in vals_flow_cnt.items()} }")
print(f"  DGap  分布: { {int(k): int(v) for k, v in vals_gap_cnt.items()} }")
print(f"  特征数量: {len(features)}")

# ── 2. 特征基本统计量 ────────────────────────────────────────────────────
print("\n【2】16个脉冲特征描述性统计（均值 / 标准差 / 最小值 / 最大值）")
print(f"  {'特征':<18} {'均值':>8} {'标准差':>8} {'最小值':>8} {'最大值':>8}")
for feat in features:
    vals = df[feat]
    print(f"  {feat:<18} {vals.mean():>8.3f} {vals.std():>8.3f} {vals.min():>8.3f} {vals.max():>8.3f}")

# ── 3. Pearson 相关系数（特征 vs DFlow / DGap） ──────────────────────────
print("\n【3】Pearson 相关系数（特征 → DFlow / DGap）")
corr_flow_vals = df[features].corrwith(df['DFlow'])
corr_gap_vals  = df[features].corrwith(df['DGap'])
print(f"  {'特征':<18} {'r(DFlow)':>10} {'r(DGap)':>10}")
for feat in features:
    print(f"  {feat:<18} {corr_flow_vals[feat]:>10.4f} {corr_gap_vals[feat]:>10.4f}")
print(f"\n  与DFlow相关性最强的前3个特征:")
top3_corr_flow = corr_flow_vals.abs().sort_values(ascending=False).head(3)
for feat, val in top3_corr_flow.items():
    sign = '+' if corr_flow_vals[feat] > 0 else '-'
    print(f"    {feat}: r = {sign}{val:.4f}")
print(f"\n  与DGap相关性最强的前3个特征:")
top3_corr_gap = corr_gap_vals.abs().sort_values(ascending=False).head(3)
for feat, val in top3_corr_gap.items():
    sign = '+' if corr_gap_vals[feat] > 0 else '-'
    print(f"    {feat}: r = {sign}{val:.4f}")

# ── 4. 特征间相关性矩阵（强相关对 |r|>0.8） ─────────────────────────────
print("\n【4】特征间强相关对（|Pearson r| > 0.8）")
corr_feat_matrix = df[features].corr()
strong_pairs = []
for i in range(len(features)):
    for j in range(i+1, len(features)):
        r = corr_feat_matrix.iloc[i, j]
        if abs(r) > 0.8:
            strong_pairs.append((features[i], features[j], r))
strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
if strong_pairs:
    for f1, f2, r in strong_pairs:
        print(f"  {f1} — {f2}: r = {r:.4f}")
else:
    print("  无强相关对（|r| > 0.8）")

# ── 5. PCA 方差解释率 ────────────────────────────────────────────────────
print("\n【5】PCA 方差解释率（前6个主成分）")
pca_v2 = PCA(n_components=6)
pca_v2.fit(X_sc)
var_exp_v2 = pca_v2.explained_variance_ratio_ * 100
cum_var_v2 = np.cumsum(var_exp_v2)
print(f"  {'PC':<6} {'方差贡献率(%)':>14} {'累计方差(%)':>12}")
for i, (v, c) in enumerate(zip(var_exp_v2, cum_var_v2)):
    print(f"  PC{i+1:<4} {v:>14.2f} {c:>12.2f}")

# 各PC的主要载荷特征（贡献最大的3个）
pca_loadings = pd.DataFrame(
    pca_v2.components_.T,
    index=features,
    columns=[f'PC{i+1}' for i in range(6)]
)
print("\n  各主成分前3大载荷特征：")
for pc in ['PC1', 'PC2', 'PC3']:
    top3 = pca_loadings[pc].abs().sort_values(ascending=False).head(3)
    loading_str = ', '.join([f"{f}({pca_loadings.loc[f, pc]:+.3f})" for f in top3.index])
    print(f"    {pc}: {loading_str}")

# ── 6. 随机森林特征重要性（全部训练集） ─────────────────────────────────
print("\n【6】随机森林特征重要性（n_estimators=200，全量数据训练）")
# 使用代码主体中已训练的 rf_flow, rf_gap（均用全量数据）
imp_flow_v2 = rf_flow.feature_importances_
imp_gap_v2  = rf_gap.feature_importances_
order_gap   = np.argsort(imp_gap_v2)[::-1]
order_flow  = np.argsort(imp_flow_v2)[::-1]

print(f"\n  DGap 特征重要性排序:")
print(f"  {'排名':<5} {'特征':<18} {'重要性(%)':>10}")
for rank, idx in enumerate(order_gap):
    print(f"  {rank+1:<5} {features[idx]:<18} {imp_gap_v2[idx]*100:>10.2f}%")

print(f"\n  DFlow 特征重要性排序:")
print(f"  {'排名':<5} {'特征':<18} {'重要性(%)':>10}")
for rank, idx in enumerate(order_flow):
    print(f"  {rank+1:<5} {features[idx]:<18} {imp_flow_v2[idx]*100:>10.2f}%")

top3_gap_names  = [features[i] for i in order_gap[:3]]
top3_flow_names = [features[i] for i in order_flow[:3]]
print(f"\n  DGap 最重要的前3个特征: {top3_gap_names}")
print(f"  DFlow 最重要的前3个特征: {top3_flow_names}")

# ── 7. 5折交叉验证——各模型 Macro F1 ─────────────────────────────────────
print(f"\n【7】5折交叉验证各模型 Macro-F1（%）")
models_v2 = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'SVM (RBF)':     SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42),
    'k-NN (k=5)':    KNeighborsClassifier(n_neighbors=5),
    'LDA':           LinearDiscriminantAnalysis(),
    'Logistic Reg.': LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42),
}
cv_v2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
print(f"  {'模型':<20} {'DFlow F1(均值)':>14} {'DFlow F1(标准差)':>16} {'DGap F1(均值)':>13} {'DGap F1(标准差)':>15}")
for name, model in models_v2.items():
    f1_flow = cross_val_score(model, X_sc, y_flow, cv=cv_v2, scoring='f1_macro') * 100
    f1_gap  = cross_val_score(model, X_sc, y_gap,  cv=cv_v2, scoring='f1_macro') * 100
    cv_results[name] = {'flow_mean': f1_flow.mean(), 'flow_std': f1_flow.std(),
                        'gap_mean':  f1_gap.mean(),  'gap_std':  f1_gap.std()}
    print(f"  {name:<20} {f1_flow.mean():>14.2f} {f1_flow.std():>16.2f} {f1_gap.mean():>13.2f} {f1_gap.std():>15.2f}")

best_flow_model = max(cv_results, key=lambda x: cv_results[x]['flow_mean'])
best_gap_model  = max(cv_results, key=lambda x: cv_results[x]['gap_mean'])
print(f"\n  DFlow 最优模型: {best_flow_model}  F1={cv_results[best_flow_model]['flow_mean']:.2f}%±{cv_results[best_flow_model]['flow_std']:.2f}%")
print(f"  DGap  最优模型: {best_gap_model}  F1={cv_results[best_gap_model]['gap_mean']:.2f}%±{cv_results[best_gap_model]['gap_std']:.2f}%")

# ── 8. 测试集详细评估（Random Forest，75/25划分） ────────────────────────
print(f"\n【8】随机森林测试集详细评估（test_size=0.25，random_state=42）")
X_tr2, X_te2, yf_tr2, yf_te2, yg_tr2, yg_te2 = train_test_split(
    X_sc, y_flow, y_gap, test_size=0.25, random_state=42, stratify=y_flow
)
rf_f2 = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
rf_g2 = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
rf_f2.fit(X_tr2, yf_tr2); rf_g2.fit(X_tr2, yg_tr2)
yf_pred2 = rf_f2.predict(X_te2); yg_pred2 = rf_g2.predict(X_te2)

acc_flow = accuracy_score(yf_te2, yf_pred2)
acc_gap  = accuracy_score(yg_te2, yg_pred2)
f1_flow_test  = f1_score(yf_te2, yf_pred2, average='macro')
f1_gap_test   = f1_score(yg_te2, yg_pred2, average='macro')
cm_flow2 = confusion_matrix(yf_te2, yf_pred2, labels=[-1, 0, 1])
cm_gap2  = confusion_matrix(yg_te2, yg_pred2, labels=[-1, 0, 1])

print(f"  训练集大小: {len(X_tr2)}，测试集大小: {len(X_te2)}")
print(f"\n  DFlow 预测结果:")
print(f"    准确率 (Accuracy) = {acc_flow*100:.2f}%")
print(f"    宏平均 F1-Score   = {f1_flow_test*100:.2f}%")
print(f"    混淆矩阵 (行=真实，列=预测，标签顺序[-1,0,1]):")
print(f"    {cm_flow2}")
print(f"\n  DFlow 分类报告:")
print(classification_report(yf_te2, yf_pred2, target_names=['DFlow=-1','DFlow=0','DFlow=1']))

print(f"\n  DGap 预测结果:")
print(f"    准确率 (Accuracy) = {acc_gap*100:.2f}%")
print(f"    宏平均 F1-Score   = {f1_gap_test*100:.2f}%")
print(f"    混淆矩阵 (行=真实，列=预测，标签顺序[-1,0,1]):")
print(f"    {cm_gap2}")
print(f"\n  DGap 分类报告:")
print(classification_report(yg_te2, yg_pred2, target_names=['DGap=-1','DGap=0','DGap=1']))

# ── 9. DFlow/DGap 联合状态分布 ──────────────────────────────────────────
print(f"\n【9】DFlow–DGap 联合状态频次分布")
combo_v2 = df.groupby(['DFlow', 'DGap']).size().unstack(fill_value=0)
print(combo_v2.to_string())
# 最高频组合
idx_max = np.unravel_index(combo_v2.values.argmax(), combo_v2.shape)
dflow_max = combo_v2.index[idx_max[0]]
dgap_max  = combo_v2.columns[idx_max[1]]
print(f"\n  最高频联合状态: DFlow={dflow_max}, DGap={dgap_max}，出现次数={combo_v2.values[idx_max]}")

# ── 10. Kruskal-Wallis 检验（特征 vs 复合指标 S） ──────────────────────
print(f"\n【10】Kruskal-Wallis 检验结果（特征 vs 复合控制指标 S=DFlow+DGap）")
df_v2 = df.copy()
df_v2['S'] = df_v2['DFlow'] + df_v2['DGap']
s_vals_v2 = sorted(df_v2['S'].unique())
print(f"  S 可能取值: {s_vals_v2}")
print(f"  {'特征':<18} {'H 统计量':>10} {'p 值':>12} {'显著性':>8}")
kw_results = {}
chi2_crit = stats.chi2.ppf(0.95, df=len(s_vals_v2)-1)
for feat in features:
    groups_v2 = [df_v2[df_v2['S'] == s][feat].values for s in s_vals_v2]
    h_v2, p_v2 = stats.kruskal(*[g for g in groups_v2 if len(g) > 0])
    kw_results[feat] = {'H': h_v2, 'p': p_v2}
    sig_label = '***' if p_v2 < 0.001 else ('**' if p_v2 < 0.01 else ('*' if p_v2 < 0.05 else 'ns'))
    print(f"  {feat:<18} {h_v2:>10.3f} {p_v2:>12.4f} {sig_label:>8}")
print(f"\n  chi2临界值 (df={len(s_vals_v2)-1}, α=0.05) = {chi2_crit:.3f}")
sig_feats = [f for f, v in kw_results.items() if v['p'] < 0.05]
print(f"  显著性特征数量 (p<0.05): {len(sig_feats)} / {len(features)}")
print(f"  显著性特征列表: {sig_feats}")

# ── 11. DFlow 与 DGap 之间的相关性及联合分析 ────────────────────────────
print(f"\n【11】DFlow–DGap 协同关系分析")
r_fd, p_fd = stats.pearsonr(df['DFlow'], df['DGap'])
rho_fd, p_rho_fd = stats.spearmanr(df['DFlow'], df['DGap'])
print(f"  Pearson r (DFlow, DGap)  = {r_fd:.4f},  p = {p_fd:.4f}")
print(f"  Spearman ρ (DFlow, DGap) = {rho_fd:.4f},  p = {p_rho_fd:.4f}")
# Chi-square 独立性检验
ct = pd.crosstab(df['DFlow'], df['DGap'])
chi2_stat, p_chi2, dof_chi2, _ = stats.chi2_contingency(ct)
print(f"  卡方检验 χ² = {chi2_stat:.4f},  df = {dof_chi2},  p = {p_chi2:.4f}")
print(f"  结论: DFlow 与 DGap {'存在' if p_chi2 < 0.05 else '不存在'}显著关联 (α=0.05)")

# ── 12. LDA 判别能力（各类均值在LD1轴上的间距） ─────────────────────────
print(f"\n【12】LDA 类别可分性分析")
lda_gap_v2  = LinearDiscriminantAnalysis()
lda_flow_v2 = LinearDiscriminantAnalysis()
X_lda_gap_v2  = lda_gap_v2.fit_transform(X_sc, y_gap)
X_lda_flow_v2 = lda_flow_v2.fit_transform(X_sc, y_flow)

print("  DGap — LD1 各类均值:")
for cls in [-1, 0, 1]:
    mask = y_gap == cls
    vals_ld = X_lda_gap_v2[mask, 0] if X_lda_gap_v2.ndim > 1 else X_lda_gap_v2[mask]
    print(f"    DGap={cls}: LD1均值 = {vals_ld.mean():.4f}, 标准差 = {vals_ld.std():.4f}")

print("  DFlow — LD1 各类均值:")
for cls in [-1, 0, 1]:
    mask = y_flow == cls
    if np.sum(mask) > 0:
        vals_ld = X_lda_flow_v2[mask, 0] if X_lda_flow_v2.ndim > 1 else X_lda_flow_v2[mask]
        print(f"    DFlow={cls}: LD1均值 = {vals_ld.mean():.4f}, 标准差 = {vals_ld.std():.4f}")

# ── 13. 各类别特征均值对比（DGap分组） ──────────────────────────────────
print(f"\n【13】各类别特征均值对比（按 DGap 分组）")
print(f"  {'特征':<18} {'DGap=-1':>10} {'DGap=0':>10} {'DGap=+1':>10} {'Kruskal p':>12}")
for feat in features:
    grp = [df[df['DGap'] == c][feat].values for c in [-1, 0, 1]]
    means = [g.mean() for g in grp]
    h_stat, p_stat = stats.kruskal(*grp)
    sig = '***' if p_stat < 0.001 else ('**' if p_stat < 0.01 else ('*' if p_stat < 0.05 else 'ns'))
    print(f"  {feat:<18} {means[0]:>10.3f} {means[1]:>10.3f} {means[2]:>10.3f} {p_stat:>10.4f}{sig:>2}")

# ── 14. 最终汇总（论文直接引用数据） ────────────────────────────────────
print(f"\n{SEP}")
print("★  论文关键数值汇总（可直接引用）")
print(SEP)
best_gap_f1    = max(cv_results[m]['gap_mean']  for m in cv_results)
best_flow_f1   = max(cv_results[m]['flow_mean'] for m in cv_results)
print(f"  数据集: {len(df)} 个样本，{len(features)} 个脉冲特征参数")
print(f"  DFlow 分布: {{-1: {int(vals_flow_cnt.get(-1,0))}, 0: {int(vals_flow_cnt.get(0,0))}, +1: {int(vals_flow_cnt.get(1,0))}}}")
print(f"  DGap  分布: {{-1: {int(vals_gap_cnt.get(-1,0))}, 0: {int(vals_gap_cnt.get(0,0))}, +1: {int(vals_gap_cnt.get(1,0))}}}")
print(f"  PCA前3个主成分累计方差解释率: {cum_var_v2[2]:.1f}%")
print(f"  DGap最重要特征（RF）: {top3_gap_names}")
print(f"  DFlow最重要特征（RF）: {top3_flow_names}")
print(f"  5折CV最优 DFlow Macro-F1: {best_flow_f1:.1f}% （{best_flow_model}）")
print(f"  5折CV最优 DGap  Macro-F1: {best_gap_f1:.1f}%  （{best_gap_model}）")
print(f"  测试集 RF DFlow 准确率: {acc_flow*100:.1f}%，F1={f1_flow_test*100:.1f}%")
print(f"  测试集 RF DGap  准确率: {acc_gap*100:.1f}%，F1={f1_gap_test*100:.1f}%")
print(f"  DFlow–DGap Pearson r = {r_fd:.4f}，Spearman ρ = {rho_fd:.4f}")
print(f"  KW检验显著特征数（vs S）: {len(sig_feats)}/{len(features)}")
print(SEP)
print("\n✓ 数值计算模块完成！")
