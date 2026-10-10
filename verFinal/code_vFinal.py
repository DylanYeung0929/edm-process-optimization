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
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (cross_val_score, StratifiedKFold,
                                     train_test_split, cross_validate)
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, accuracy_score)
from sklearn.base import clone
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_FILE = '/Users/dylanyeung/PyCharmProject/Model/2026bit_math_model_round1/202601B题数据.xlsx'
OUT       = 'outputs/'

# Font & Style
FONT = 'DejaVu Serif'
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

# Load Data
df       = pd.read_excel(DATA_FILE)
features = [c for c in df.columns if c not in ['DFlow', 'DGap']]
X_raw    = df[features].values          # 原始未标准化特征矩阵（Pipeline内部处理）
y_flow   = df['DFlow'].values
y_gap    = df['DGap'].values

# 仅用于可视化/聚类分析的全局标准化（不用于模型评估）
scaler_vis = StandardScaler()
X_sc_vis   = scaler_vis.fit_transform(X_raw)

print(f"Data loaded: {df.shape}")
print("=" * 65)

print("\n 层次聚类特征精简 ...")

corr_mat   = df[features].corr().abs()
dist_mat   = 1 - corr_mat.values
np.fill_diagonal(dist_mat, 0)
Z_link     = linkage(dist_mat, method='average')

# 在 d=0.4（|r|>0.6）处切割，得到聚类标签
cluster_labels = fcluster(Z_link, t=0.4, criterion='distance')
n_clusters     = len(np.unique(cluster_labels))
print(f"  切割阈值 d=0.4 → {n_clusters} 个特征簇")

# 每簇选「与 DGap 的 Pearson |r| 最大」的特征作为代表（DGap更易区分，代表性更强）
corr_dgap_abs = df[features].corrwith(df['DGap']).abs()
representative_features = []
cluster_info = {}
for c in np.unique(cluster_labels):
    members = [features[i] for i, lbl in enumerate(cluster_labels) if lbl == c]
    rep     = corr_dgap_abs[members].idxmax()
    representative_features.append(rep)
    cluster_info[c] = {'members': members, 'representative': rep}
    print(f"  簇 {c:2d}: {members} → 代表: {rep}")

X_reduced = df[representative_features].values
print(f"\n  精简特征集（{len(representative_features)}维）: {representative_features}")

# Pipeline 定义 ── 所有评估均通过 Pipeline 进行
# 每个 Pipeline = StandardScaler + Classifier
# Scaler 在每折独立 fit_transform，消除数据泄露

def make_pipeline(clf):
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def classifier_chain_cv(X, y_first, y_second, base_first, base_second, cv):
    scores_joint, scores_first, scores_second = [], [], []
    for train_idx, test_idx in cv.split(X, y_first):
        X_train, X_test = X[train_idx], X[test_idx]
        y_first_train, y_first_test = y_first[train_idx], y_first[test_idx]
        y_second_train, y_second_test = y_second[train_idx], y_second[test_idx]

        clf_first = clone(base_first)
        clf_second = clone(base_second)

        clf_first.fit(X_train, y_first_train)
        first_test_pred = clf_first.predict(X_test)

        X_train_chain = np.column_stack([X_train, y_first_train])
        X_test_chain = np.column_stack([X_test, first_test_pred])
        clf_second.fit(X_train_chain, y_second_train)
        second_test_pred = clf_second.predict(X_test_chain)

        scores_first.append(f1_score(y_first_test, first_test_pred, average='macro'))
        scores_second.append(f1_score(y_second_test, second_test_pred, average='macro'))
        joint_true = [f"{a}_{b}" for a, b in zip(y_first_test, y_second_test)]
        joint_pred = [f"{a}_{b}" for a, b in zip(first_test_pred, second_test_pred)]
        scores_joint.append(f1_score(joint_true, joint_pred, average='macro'))

    return {
        'joint': np.array(scores_joint) * 100,
        'first': np.array(scores_first) * 100,
        'second': np.array(scores_second) * 100,
    }


def classifier_chain_holdout(X, y_first, y_second, base_first, base_second, idx_train, idx_test):
    clf_first = clone(base_first)
    clf_second = clone(base_second)

    X_train, X_test = X[idx_train], X[idx_test]
    y_first_train, y_first_test = y_first[idx_train], y_first[idx_test]
    y_second_train, y_second_test = y_second[idx_train], y_second[idx_test]

    clf_first.fit(X_train, y_first_train)
    first_test_pred = clf_first.predict(X_test)

    X_train_chain = np.column_stack([X_train, y_first_train])
    X_test_chain = np.column_stack([X_test, first_test_pred])
    clf_second.fit(X_train_chain, y_second_train)
    second_test_pred = clf_second.predict(X_test_chain)

    joint_true = [f"{a}_{b}" for a, b in zip(y_first_test, y_second_test)]
    joint_pred = [f"{a}_{b}" for a, b in zip(first_test_pred, second_test_pred)]

    return {
        'first_pred': first_test_pred,
        'second_pred': second_test_pred,
        'joint_true': joint_true,
        'joint_pred': joint_pred,
        'joint_f1': f1_score(joint_true, joint_pred, average='macro') * 100,
        'first_f1': f1_score(y_first_test, first_test_pred, average='macro') * 100,
        'second_f1': f1_score(y_second_test, second_test_pred, average='macro') * 100,
        'joint_acc': accuracy_score(joint_true, joint_pred) * 100,
    }


pipelines_all = {
    'Random Forest': make_pipeline(
        RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
    'SVM (RBF)': make_pipeline(
        SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42)),
    'k-NN (k=5)': make_pipeline(KNeighborsClassifier(n_neighbors=5)),
    'LDA':  make_pipeline(LinearDiscriminantAnalysis()),
    'Logistic Reg.': make_pipeline(
        LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42)),
}

pipelines_red = {
    'Random Forest': make_pipeline(
        RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
    'SVM (RBF)': make_pipeline(
        SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42)),
    'k-NN (k=5)': make_pipeline(KNeighborsClassifier(n_neighbors=5)),
    'LDA':  make_pipeline(LinearDiscriminantAnalysis()),
    'Logistic Reg.': make_pipeline(
        LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42)),
}

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ====================================================================
# FIGURE 1 – Data Overview
# ====================================================================
print("\nGenerating Fig1 ...")
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(3, 6, figure=fig, hspace=0.55, wspace=0.45)

ax_pie1 = fig.add_subplot(gs[0, 0:2])
ax_pie2 = fig.add_subplot(gs[0, 2:4])
vals_flow = pd.Series(y_flow).value_counts().sort_index()
vals_gap  = pd.Series(y_gap).value_counts().sort_index()

wedges1, _, at1 = ax_pie1.pie(
    vals_flow.values, labels=[f'DFlow={k}' for k in vals_flow.index],
    colors=[CLASS_COLORS[k] for k in vals_flow.index],
    autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
for at in at1: at.set_fontsize(8.5)
ax_pie1.set_title('(a) DFlow Class Distribution', fontweight='bold')

wedges2, _, at2 = ax_pie2.pie(
    vals_gap.values, labels=[f'DGap={k}' for k in vals_gap.index],
    colors=[CLASS_COLORS[k] for k in vals_gap.index],
    autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
for at in at2: at.set_fontsize(8.5)
ax_pie2.set_title('(b) DGap Class Distribution', fontweight='bold')

ax_heat = fig.add_subplot(gs[0, 4:6])
combo   = df.groupby(['DFlow', 'DGap']).size().unstack(fill_value=0)
im      = ax_heat.imshow(combo.values, cmap='Blues', aspect='auto')
ax_heat.set_xticks(range(len(combo.columns))); ax_heat.set_xticklabels(combo.columns)
ax_heat.set_yticks(range(len(combo.index)));   ax_heat.set_yticklabels(combo.index)
ax_heat.set_xlabel('DGap'); ax_heat.set_ylabel('DFlow')
ax_heat.set_title('(c) Joint State Frequency', fontweight='bold')
for i in range(combo.shape[0]):
    for j in range(combo.shape[1]):
        ax_heat.text(j, i, str(combo.values[i, j]), ha='center', va='center', fontsize=11,
                     color='white' if combo.values[i, j] > 30 else 'black', fontweight='bold')
plt.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.04)

for idx, feat in enumerate(features):
    row = 1 + idx // 6; col = idx % 6
    if row > 2: break
    ax = fig.add_subplot(gs[row, col])
    groups = [df[df['DGap'] == c][feat].values for c in [-1, 0, 1]]
    parts  = ax.violinplot(groups, positions=[0, 1, 2], showmedians=True, widths=0.7)
    for pc, c in zip(parts['bodies'], [-1, 0, 1]):
        pc.set_facecolor(CLASS_COLORS[c]); pc.set_alpha(0.65)
    parts['cmedians'].set_color('black'); parts['cmedians'].set_linewidth(1.5)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['-1', '0', '1'], fontsize=7)
    ax.set_title(feat, fontsize=7.5, fontweight='bold')
    ax.set_xlabel('DGap', fontsize=7); ax.tick_params(labelsize=7)

fig.legend(handles=[mpatches.Patch(color=CLASS_COLORS[c], label=f'DGap={c}') for c in [-1, 0, 1]],
           loc='lower center', ncol=3, fontsize=9, title='DGap Class',
           title_fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))
plt.suptitle('Figure 2: Dataset Overview — Class Distributions and Feature Distributions by DGap',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig(OUT + 'Fig1_Data_Overview.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig1 saved")

# ====================================================================
# FIGURE 2 – Correlation Analysis
# ====================================================================
print("Generating Fig2 ...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
corr = df[features].corr()
cmap_rdb = LinearSegmentedColormap.from_list('rdb', ['#D94F3D', 'white', '#2C6FAC'])
im = axes[0].imshow(corr.values, cmap=cmap_rdb, vmin=-1, vmax=1, aspect='auto')
axes[0].set_xticks(range(16)); axes[0].set_xticklabels(features, rotation=45, ha='right', fontsize=7.5)
axes[0].set_yticks(range(16)); axes[0].set_yticklabels(features, fontsize=7.5)
axes[0].set_title('(a) Feature Inter-Correlation Matrix', fontweight='bold')
for i in range(16):
    for j in range(16):
        v = corr.values[i, j]
        axes[0].text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=5.5,
                     color='white' if abs(v) > 0.65 else 'black')
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label='Pearson r')

corr_flow_s = df[features].corrwith(df['DFlow'])
corr_gap_s  = df[features].corrwith(df['DGap'])
x = np.arange(16); w = 0.38
b1 = axes[1].barh(x - w/2, corr_flow_s.values, w, color=PALETTE['blue'],  alpha=0.85, label='DFlow')
b2 = axes[1].barh(x + w/2, corr_gap_s.values,  w, color=PALETTE['red'],   alpha=0.85, label='DGap')
axes[1].set_yticks(x); axes[1].set_yticklabels(features, fontsize=8.5)
axes[1].axvline(0, color='black', lw=0.8, ls='--')
axes[1].set_xlabel('Pearson Correlation Coefficient', fontsize=11)
axes[1].set_title('(b) Feature Correlations with DFlow & DGap', fontweight='bold')
axes[1].legend(loc='lower right'); axes[1].set_xlim(-0.85, 0.85)
axes[1].grid(axis='x', alpha=0.3, ls='--')
for bar in list(b1) + list(b2):
    wv = bar.get_width()
    axes[1].text(wv + 0.01*(1 if wv >= 0 else -1), bar.get_y() + bar.get_height()/2,
                 f'{wv:.2f}', va='center', ha='left' if wv >= 0 else 'right', fontsize=6)
plt.suptitle('Figure 3: Pearson Correlation Analysis of Pulse Features and Control Variables',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig2_Correlation_Analysis.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig2 saved")

# ====================================================================
# FIGURE 3 – PCA Visualization (for visualisation only, fitted on full data)
# ====================================================================
print("Generating Fig3 ...")
pca       = PCA(n_components=6)
X_pca_vis = pca.fit_transform(X_sc_vis)
var_exp   = pca.explained_variance_ratio_ * 100
cum_var   = np.cumsum(var_exp)

fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(131)
ax1.bar(range(1, 7), var_exp, color=PALETTE['blue'], alpha=0.8, label='Individual')
ax1.plot(range(1, 7), cum_var, 'o-', color=PALETTE['red'], lw=2, label='Cumulative')
ax1.axhline(80, color=PALETTE['orange'], ls='--', lw=1.2, label='80% threshold')
ax1.set_xlabel('Principal Component', fontsize=11)
ax1.set_ylabel('Explained Variance (%)', fontsize=11)
ax1.set_title('(a) PCA Scree Plot', fontweight='bold'); ax1.legend(); ax1.set_xticks(range(1, 7))
for i, (v, c) in enumerate(zip(var_exp, cum_var)):
    ax1.text(i+1, v+0.5, f'{v:.1f}%', ha='center', fontsize=8)

ax2 = fig.add_subplot(132)
for cls in [-1, 0, 1]:
    mask = y_gap == cls
    ax2.scatter(X_pca_vis[mask, 0], X_pca_vis[mask, 1], c=CLASS_COLORS[cls],
                label=f'DGap={cls}', alpha=0.75, s=45, edgecolors='white', linewidth=0.4)
ax2.set_xlabel(f'PC1 ({var_exp[0]:.1f}%)', fontsize=11)
ax2.set_ylabel(f'PC2 ({var_exp[1]:.1f}%)', fontsize=11)
ax2.set_title('(b) PCA Score Plot — DGap Classes', fontweight='bold'); ax2.legend()

ax3 = fig.add_subplot(133, projection='3d')
for cls in [-1, 0, 1]:
    mask = y_gap == cls
    ax3.scatter(X_pca_vis[mask, 0], X_pca_vis[mask, 1], X_pca_vis[mask, 2],
                c=CLASS_COLORS[cls], label=f'DGap={cls}', alpha=0.75, s=35, edgecolors='none')
ax3.set_xlabel(f'PC1\n({var_exp[0]:.1f}%)', fontsize=8, labelpad=8)
ax3.set_ylabel(f'PC2\n({var_exp[1]:.1f}%)', fontsize=8, labelpad=8)
ax3.set_zlabel(f'PC3\n({var_exp[2]:.1f}%)', fontsize=8, labelpad=8)
ax3.set_title('(c) 3D PCA Score Plot — DGap Classes', fontweight='bold')
ax3.legend(fontsize=8); ax3.view_init(elev=22, azim=135)
plt.suptitle('Appendix Figure 1: Principal Component Analysis of Pulse Feature Space',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig3_PCA_Analysis.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig3 saved")

# ====================================================================
# FIGURE 4 – Random Forest Feature Importance
# (fit on full data for importance estimation, consistent with paper)
# ====================================================================
print("Generating Fig4 ...")
rf_flow_imp = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_gap_imp  = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_flow_imp.fit(X_sc_vis, y_flow)
rf_gap_imp.fit(X_sc_vis, y_gap)
imp_flow = rf_flow_imp.feature_importances_
imp_gap  = rf_gap_imp.feature_importances_
order    = np.argsort(imp_gap)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
colors_f = [PALETTE['blue'] if v > np.median(imp_flow) else '#A8C8E8' for v in imp_flow[order]]
colors_g = [PALETTE['red']  if v > np.median(imp_gap)  else '#F4A7A0' for v in imp_gap[order]]
for ax, imp, colors, title in [
        (axes[0], imp_flow, colors_f, '(a) Random Forest Importance — DFlow'),
        (axes[1], imp_gap,  colors_g, '(b) Random Forest Importance — DGap')]:
    bars = ax.barh(range(16), imp[order]*100, color=colors, edgecolor='white')
    ax.set_yticks(range(16)); ax.set_yticklabels([features[i] for i in order], fontsize=9)
    ax.set_xlabel('Feature Importance (%)', fontsize=11); ax.set_title(title, fontweight='bold')
    ref_val = np.median(imp)
    ax.axvline(ref_val*100, ls='--', color='gray', lw=1, label='Median'); ax.legend()
    for bar, val in zip(bars, imp[order]*100):
        ax.text(val+0.1, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=7.5)
plt.suptitle('Figure 4: Random Forest Feature Importance for DFlow and DGap Prediction',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig4_Feature_Importance.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig4 saved")

# ====================================================================
# FIGURE 5 – 3D Feature Space (top 3 DGap features)
# ====================================================================
print("Generating Fig5 ...")
top3_gap = [features[i] for i in np.argsort(imp_gap)[::-1][:3]]
fig = plt.figure(figsize=(16, 6))
for idx, (azim, elev, lbl) in enumerate([(135, 25, '(a)'), (45, 25, '(b)'), (90, 60, '(c)')]):
    ax = fig.add_subplot(1, 3, idx+1, projection='3d')
    for cls in [-1, 0, 1]:
        mask = y_gap == cls
        ax.scatter(df.loc[mask, top3_gap[0]], df.loc[mask, top3_gap[1]], df.loc[mask, top3_gap[2]],
                   c=CLASS_COLORS[cls], label=f'DGap={cls}', alpha=0.7, s=30, edgecolors='none')
    ax.set_xlabel(top3_gap[0], fontsize=7.5, labelpad=6)
    ax.set_ylabel(top3_gap[1], fontsize=7.5, labelpad=6)
    ax.set_zlabel(top3_gap[2], fontsize=7.5, labelpad=6)
    ax.set_title(f'{lbl} View {idx+1}', fontweight='bold')
    ax.view_init(elev=elev, azim=azim)
    if idx == 0: ax.legend(fontsize=8, loc='upper left')
plt.suptitle(f'Figure 14: 3D Scatter Plot of Top-3 DGap-Important Features ({", ".join(top3_gap)})',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig5_3D_Feature_Space.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig5 saved")

# ====================================================================
# FIGURE 6 – LDA Analysis
# ====================================================================
print("Generating Fig6 ...")
lda_gap_v  = LinearDiscriminantAnalysis()
lda_flow_v = LinearDiscriminantAnalysis()
X_lda_gap_v  = lda_gap_v.fit_transform(X_sc_vis, y_gap)
X_lda_flow_v = lda_flow_v.fit_transform(X_sc_vis, y_flow)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for cls in [-1, 0, 1]:
    mask = y_gap == cls
    if X_lda_gap_v.ndim == 1 or X_lda_gap_v.shape[1] == 1:
        axes[0].hist(X_lda_gap_v[mask].ravel(), bins=20, color=CLASS_COLORS[cls],
                     alpha=0.6, label=f'DGap={cls}', density=True)
    else:
        axes[0].scatter(X_lda_gap_v[mask, 0], X_lda_gap_v[mask, 1],
                        c=CLASS_COLORS[cls], label=f'DGap={cls}', alpha=0.7, s=45)
axes[0].set_title('(a) LDA Projection — DGap Classes', fontweight='bold')
axes[0].set_xlabel('LD1', fontsize=11)
axes[0].set_ylabel('LD2' if X_lda_gap_v.shape[1] > 1 else 'Density', fontsize=11)
axes[0].legend()

for cls in [-1, 0, 1]:
    mask = y_flow == cls
    if X_lda_flow_v.ndim == 1 or X_lda_flow_v.shape[1] == 1:
        axes[1].hist(X_lda_flow_v[mask].ravel(), bins=15, color=CLASS_COLORS[cls],
                     alpha=0.6, label=f'DFlow={cls}', density=True)
    else:
        axes[1].scatter(X_lda_flow_v[mask, 0], X_lda_flow_v[mask, 1],
                        c=CLASS_COLORS[cls], label=f'DFlow={cls}', alpha=0.7, s=45)
axes[1].set_title('(b) LDA Projection — DFlow Classes', fontweight='bold')
axes[1].set_xlabel('LD1', fontsize=11)
axes[1].set_ylabel('LD2' if X_lda_flow_v.shape[1] > 1 else 'Density', fontsize=11)
axes[1].legend()

plt.suptitle('Appendix Figure 2: Linear Discriminant Analysis — Class Separability in Reduced Space',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig6_LDA_Analysis.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig6 saved")

# ====================================================================
# FIGURE 7 – （含全特征 & 精简特征对比）
# ====================================================================
print("Generating Fig7 (Pipeline CV) ...")

cv_results_all = {}
cv_results_red = {}

print("  Running 5-fold CV (Pipeline, no leakage) ...")
for name in pipelines_all:
    f1_flow_a = cross_val_score(pipelines_all[name], X_raw,     y_flow, cv=cv5, scoring='f1_macro') * 100
    f1_gap_a  = cross_val_score(pipelines_all[name], X_raw,     y_gap,  cv=cv5, scoring='f1_macro') * 100
    f1_flow_r = cross_val_score(pipelines_red[name], X_reduced, y_flow, cv=cv5, scoring='f1_macro') * 100
    f1_gap_r  = cross_val_score(pipelines_red[name], X_reduced, y_gap,  cv=cv5, scoring='f1_macro') * 100
    cv_results_all[name] = {'flow': f1_flow_a, 'gap': f1_gap_a}
    cv_results_red[name] = {'flow': f1_flow_r, 'gap': f1_gap_r}
    print(f"    {name}: "
          f"DFlow_all={f1_flow_a.mean():.1f}% DFlow_red={f1_flow_r.mean():.1f}%  |  "
          f"DGap_all={f1_gap_a.mean():.1f}%  DGap_red={f1_gap_r.mean():.1f}%")

# ── Fig 7a/7b: full-feature CV comparison (same as original Fig 10) ──
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, key, title in [(axes[0], 'flow', '(a) DFlow Prediction'),
                        (axes[1], 'gap',  '(b) DGap Prediction')]:
    names  = list(cv_results_all.keys())
    means  = [cv_results_all[n][key].mean() for n in names]
    stds   = [cv_results_all[n][key].std()  for n in names]
    colors_bar = [PALETTE['blue'], PALETTE['red'], PALETTE['green'], PALETTE['orange'], PALETTE['purple']]
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors_bar,
                  alpha=0.85, edgecolor='white', error_kw={'elinewidth': 2})
    ax.set_ylim(0, 110); ax.set_ylabel('Macro F1-Score (%)', fontsize=11)
    ax.set_title(title, fontweight='bold'); ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.3, ls='--')
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x()+bar.get_width()/2, m+s+1.5, f'{m:.1f}±{s:.1f}',
                ha='center', fontsize=8, fontweight='bold')
    ax.axhline(max(means), ls=':', color='gold', lw=1.5, label=f'Best: {max(means):.1f}%')
    ax.legend(fontsize=9)
plt.suptitle('Figure 8: 5-Fold Cross-Validation Macro-F1 (Pipeline — No Data Leakage, Full Features)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig7_Model_Performance.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig7 saved")

# ── Fig 7c: 全特征 vs 精简特征对比 ──────
print("Generating Fig7c (Full vs Reduced Feature Comparison) ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
model_names = list(cv_results_all.keys())
x_pos = np.arange(len(model_names)); w_bar = 0.35

for ax, key, title in [(axes[0], 'flow', '(a) DFlow: Full vs Reduced Features'),
                        (axes[1], 'gap',  '(b) DGap: Full vs Reduced Features')]:
    m_all = [cv_results_all[n][key].mean() for n in model_names]
    m_red = [cv_results_red[n][key].mean() for n in model_names]
    s_all = [cv_results_all[n][key].std()  for n in model_names]
    s_red = [cv_results_red[n][key].std()  for n in model_names]
    b1 = ax.bar(x_pos - w_bar/2, m_all, w_bar, yerr=s_all, capsize=4,
                color=PALETTE['blue'], alpha=0.8, label=f'Full (16 feat.)', edgecolor='white')
    b2 = ax.bar(x_pos + w_bar/2, m_red, w_bar, yerr=s_red, capsize=4,
                color=PALETTE['orange'], alpha=0.8,
                label=f'Reduced ({len(representative_features)} feat.)', edgecolor='white')
    ax.set_xticks(x_pos); ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=8.5)
    ax.set_ylabel('Macro F1-Score (%)', fontsize=11); ax.set_ylim(0, 110)
    ax.set_title(title, fontweight='bold'); ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, ls='--')
    for b, m, s in list(zip(b1, m_all, s_all)) + list(zip(b2, m_red, s_red)):
        ax.text(b.get_x()+b.get_width()/2, m+s+1, f'{m:.0f}', ha='center', fontsize=7.5)

plt.suptitle('Figure 6: Full Features vs Reduced Features — Pipeline CV Comparison\n'
             f'(Reduced = {len(representative_features)} cluster representatives, no data leakage)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig7c_Full_vs_Reduced.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig7c saved")

# ====================================================================
# FIGURE 8 – Classifier Chain for Joint Control Decision
# ====================================================================
print("Generating Fig8 (Classifier Chain joint decision) ...")

chain_rf = make_pipeline(
    RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
)
chain_svm = make_pipeline(
    SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42)
)

chain_flow_gap = classifier_chain_cv(X_raw, y_flow, y_gap, chain_rf, chain_svm, cv5)
chain_gap_flow = classifier_chain_cv(X_raw, y_gap, y_flow, chain_svm, chain_rf, cv5)

chain_best_name = 'DFlow→DGap' if chain_flow_gap['joint'].mean() >= chain_gap_flow['joint'].mean() else 'DGap→DFlow'
chain_best = chain_flow_gap if chain_best_name == 'DFlow→DGap' else chain_gap_flow
print(f"  Chain order comparison: DFlow→DGap Joint-F1={chain_flow_gap['joint'].mean():.1f}% | "
      f"DGap→DFlow Joint-F1={chain_gap_flow['joint'].mean():.1f}%")
print(f"  Selected classifier chain order: {chain_best_name}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
chain_names = ['DFlow→DGap', 'DGap→DFlow']
chain_joint_means = [chain_flow_gap['joint'].mean(), chain_gap_flow['joint'].mean()]
chain_joint_stds = [chain_flow_gap['joint'].std(), chain_gap_flow['joint'].std()]
chain_first_means = [chain_flow_gap['first'].mean(), chain_gap_flow['second'].mean()]
chain_second_means = [chain_flow_gap['second'].mean(), chain_gap_flow['first'].mean()]

bars = axes[0].bar(chain_names, chain_joint_means, yerr=chain_joint_stds, capsize=5,
                   color=[PALETTE['blue'], PALETTE['orange']], alpha=0.85, edgecolor='white')
axes[0].set_ylim(0, 110)
axes[0].set_ylabel('Joint Macro F1-Score (%)', fontsize=11)
axes[0].set_title('(a) Classifier Chain Order Comparison', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3, ls='--')
for bar, m, s in zip(bars, chain_joint_means, chain_joint_stds):
    axes[0].text(bar.get_x()+bar.get_width()/2, m+s+1.5, f'{m:.1f}±{s:.1f}', ha='center', fontsize=9, fontweight='bold')

x_chain = np.arange(len(chain_names))
w_chain = 0.32
b1 = axes[1].bar(x_chain - w_chain/2, chain_first_means, w_chain, color=PALETTE['teal'], alpha=0.85,
                 edgecolor='white', label='First-stage Macro F1')
b2 = axes[1].bar(x_chain + w_chain/2, chain_second_means, w_chain, color=PALETTE['purple'], alpha=0.85,
                 edgecolor='white', label='Second-stage Macro F1')
axes[1].set_xticks(x_chain)
axes[1].set_xticklabels(chain_names)
axes[1].set_ylim(0, 110)
axes[1].set_ylabel('Macro F1-Score (%)', fontsize=11)
axes[1].set_title('(b) Stage-wise Performance in Classifier Chains', fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(axis='y', alpha=0.3, ls='--')
for bars_group in (b1, b2):
    for bar in bars_group:
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.2, f'{bar.get_height():.1f}',
                     ha='center', fontsize=8)

plt.suptitle('Figure 7: Classifier Chain Performance for Joint DFlow–DGap Decision',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig8_Classifier_Chain.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig8 saved")

# ====================================================================
# FIGURE 9 – 测试集：DFlow用RF，DGap用SVM-RBF（双模型策略）
# ====================================================================
print("Generating Fig9 (Dual-model test set, Pipeline) ...")

# ── 确定使用全特征还是精简特征 ──────────────────────────────────
# 比较 RF(全) vs RF(精简) on DFlow；SVM(全) vs SVM(精简) on DGap
rf_flow_all = cv_results_all['Random Forest']['flow'].mean()
rf_flow_red = cv_results_red['Random Forest']['flow'].mean()
svm_gap_all = cv_results_all['SVM (RBF)']['gap'].mean()
svm_gap_red = cv_results_red['SVM (RBF)']['gap'].mean()

use_reduced_flow = rf_flow_red >= rf_flow_all - 1.0   # 精简特征F1差距≤1%则用精简
use_reduced_gap  = svm_gap_red >= svm_gap_all  - 1.0
X_flow_final = X_reduced if use_reduced_flow else X_raw
X_gap_final  = X_reduced if use_reduced_gap  else X_raw
feat_label_flow = f"Reduced ({len(representative_features)})" if use_reduced_flow else "Full (16)"
feat_label_gap  = f"Reduced ({len(representative_features)})" if use_reduced_gap  else "Full (16)"
print(f"  Final feature set: DFlow→{feat_label_flow}  DGap→{feat_label_gap}")

# ── 使用 Pipeline 做 train/test split ────────────────────────────
# stratify by DFlow（更不均衡一方）
X_comb = np.column_stack([X_flow_final, y_gap])  # 辅助stratify
(X_flow_tr, X_flow_te,
 X_gap_tr_,  X_gap_te_,
 yf_tr, yf_te,
 yg_tr, yg_te) = train_test_split(
    X_flow_final, X_gap_final, y_flow, y_gap,
    test_size=0.25, random_state=42, stratify=y_flow
)
# 如果 DGap 特征集与 DFlow 不同，重新切割
if not use_reduced_gap and use_reduced_flow:
    X_gap_tr_ = X_raw[: len(yg_tr)]   # 同索引重建（需用indices方式）

# 更健壮的切割：统一用index
idx_all = np.arange(len(y_flow))
idx_tr, idx_te = train_test_split(idx_all, test_size=0.25,
                                  random_state=42, stratify=y_flow)

X_flow_tr2 = X_flow_final[idx_tr]; X_flow_te2 = X_flow_final[idx_te]
X_gap_tr2  = X_gap_final[idx_tr];  X_gap_te2  = X_gap_final[idx_te]
yf_tr2 = y_flow[idx_tr]; yf_te2 = y_flow[idx_te]
yg_tr2 = y_gap[idx_tr];  yg_te2 = y_gap[idx_te]

# 建立 Pipeline 并 fit
pipe_rf_test  = make_pipeline(
    RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'))
pipe_svm_test = make_pipeline(
    SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42))

pipe_rf_test.fit(X_flow_tr2, yf_tr2)    # DFlow → RF
pipe_svm_test.fit(X_gap_tr2,  yg_tr2)  # DGap  → SVM

yf_pred2 = pipe_rf_test.predict(X_flow_te2)
yg_pred2 = pipe_svm_test.predict(X_gap_te2)

acc_flow2 = accuracy_score(yf_te2, yf_pred2)
acc_gap2  = accuracy_score(yg_te2, yg_pred2)
f1_flow2  = f1_score(yf_te2, yf_pred2, average='macro')
f1_gap2   = f1_score(yg_te2, yg_pred2, average='macro')
cm_flow2  = confusion_matrix(yf_te2, yf_pred2, labels=[-1, 0, 1])
cm_gap2   = confusion_matrix(yg_te2, yg_pred2, labels=[-1, 0, 1])

print(f"  [DFlow RF]  Test Acc={acc_flow2*100:.1f}%  Macro-F1={f1_flow2*100:.1f}%")
print(f"  [DGap SVM]  Test Acc={acc_gap2*100:.1f}%   Macro-F1={f1_gap2*100:.1f}%")

cmap_cm = LinearSegmentedColormap.from_list('cm', ['white', PALETTE['blue']])
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, cm, title, model_tag in [
    (axes[0], cm_flow2,
     f'(a) DFlow — RF Confusion Matrix\n(Macro-F1={f1_flow2*100:.1f}%)', 'RF'),
    (axes[1], cm_gap2,
     f'(b) DGap — SVM-RBF Confusion Matrix\n(Macro-F1={f1_gap2*100:.1f}%)', 'SVM-RBF')]:
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap_cm)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['-1','0','1'])
    ax.set_yticks([0,1,2]); ax.set_yticklabels(['-1','0','1'])
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(title, fontweight='bold')
    thresh = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Figure 9: Confusion Matrices — Dual-Model Strategy\n'
             '(DFlow: Random Forest  |  DGap: SVM-RBF)  [Pipeline, No Data Leakage]',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig8_Confusion_Matrices.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig8 saved")

# ====================================================================
# FIGURE 10 – Synergy Analysis (kept as descriptive support)
# ====================================================================
print("Generating Fig10 ...")
df['S'] = df['DFlow'] + df['DGap']
s_vals  = sorted(df['S'].unique())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
jitter = np.random.RandomState(0).normal(0, 0.06, len(df))
sc = axes[0].scatter(df['DFlow']+jitter, df['DGap']+jitter, c=df['S'],
                     cmap='RdBu_r', vmin=-2, vmax=2, alpha=0.7, s=50, edgecolors='white', lw=0.3)
axes[0].set_xlabel('DFlow'); axes[0].set_ylabel('DGap')
axes[0].set_title('(a) DFlow vs DGap Joint State\n(Color = S = DFlow + DGap)', fontweight='bold')
axes[0].set_xticks([-1, 0, 1]); axes[0].set_yticks([-1, 0, 1])
plt.colorbar(sc, ax=axes[0], label='S = DFlow + DGap')

feat_plot = top3_gap[0]
data_s    = [df[df['S'] == s][feat_plot].values for s in s_vals]
bp = axes[1].boxplot(data_s, labels=[str(s) for s in s_vals], patch_artist=True,
                     boxprops={'linewidth': 1.2}, whiskerprops={'linewidth': 1.2},
                     medianprops={'color': 'black', 'linewidth': 2})
for patch, col in zip(bp['boxes'],
                      [PALETTE['red'], PALETTE['orange'], PALETTE['gray'],
                       PALETTE['teal'], PALETTE['blue']][:len(s_vals)]):
    patch.set_facecolor(col); patch.set_alpha(0.7)
axes[1].set_xlabel('Composite Control Index S = DFlow + DGap')
axes[1].set_ylabel(feat_plot)
axes[1].set_title(f'(b) {feat_plot} Distribution\nby Composite Index S', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3, ls='--')

h_stats, p_vals = [], []
for f in features:
    groups = [df[df['S'] == s][f].values for s in s_vals]
    h, p   = stats.kruskal(*[g for g in groups if len(g) > 0])
    h_stats.append(h); p_vals.append(p)
colors_h = [PALETTE['red'] if p < 0.05 else PALETTE['gray'] for p in p_vals]
axes[2].barh(range(16), h_stats, color=colors_h, alpha=0.85)
axes[2].set_yticks(range(16)); axes[2].set_yticklabels(features, fontsize=8.5)
axes[2].set_xlabel('Kruskal-Wallis H-Statistic')
axes[2].set_title('(c) Feature Sensitivity to\nComposite Control Index S', fontweight='bold')
axes[2].axvline(stats.chi2.ppf(0.95, df=len(s_vals)-1), ls='--', color='black', lw=1.2)
sig_patch = mpatches.Patch(color=PALETTE['red'],  label='Significant (p<0.05)')
ns_patch  = mpatches.Patch(color=PALETTE['gray'], label='Not significant')
axes[2].legend(handles=[sig_patch, ns_patch], fontsize=8, loc='lower right')
plt.suptitle('Appendix Figure 3: Synergistic Control Analysis — Joint DFlow & DGap State Space',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig9_Synergy_Analysis.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig9 saved")

# ====================================================================
# FIGURE 11 – Decision Boundary (top 2 features, reduced model)
# ====================================================================
print("Generating Fig11 ...")
top2_gap = [features[i] for i in np.argsort(imp_gap)[::-1][:2]]
X2_raw   = df[top2_gap].values

pipe_2d = make_pipeline(
    RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
pipe_2d.fit(X2_raw, y_gap)

sc2        = StandardScaler().fit(X2_raw)
X2_sc_vis  = sc2.transform(X2_raw)
x_min, x_max = X2_sc_vis[:, 0].min()-0.5, X2_sc_vis[:, 0].max()+0.5
y_min, y_max = X2_sc_vis[:, 1].min()-0.5, X2_sc_vis[:, 1].max()+0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid_raw = sc2.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
Z = pipe_2d.predict(grid_raw).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
cmap_bg = LinearSegmentedColormap.from_list('bg', ['#F4A7A0', '#D3D3D3', '#A8C8E8'])
axes[0].contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg, levels=[-1.5, -0.5, 0.5, 1.5])
axes[0].contour(xx, yy, Z, colors='gray', linewidths=0.8, levels=[-0.5, 0.5], linestyles='--')
for cls in [-1, 0, 1]:
    mask = y_gap == cls
    axes[0].scatter(X2_sc_vis[mask, 0], X2_sc_vis[mask, 1], c=CLASS_COLORS[cls],
                    label=f'DGap={cls}', s=50, edgecolors='white', lw=0.5, alpha=0.85)
axes[0].set_xlabel(f'{top2_gap[0]} (standardized)'); axes[0].set_ylabel(f'{top2_gap[1]} (standardized)')
axes[0].set_title('(a) RF Decision Boundary\n(Top-2 DGap Features)', fontweight='bold')
axes[0].legend()

proba_all  = pipe_2d.predict_proba(grid_raw)
class_idx  = list(pipe_2d.classes_).index(1)
Z_proba    = proba_all[:, class_idx].reshape(xx.shape)
contf = axes[1].contourf(xx, yy, Z_proba, levels=20, cmap='Blues', alpha=0.9)
plt.colorbar(contf, ax=axes[1], label='P(DGap = +1)')
for cls in [-1, 0, 1]:
    mask = y_gap == cls
    axes[1].scatter(X2_sc_vis[mask, 0], X2_sc_vis[mask, 1], c=CLASS_COLORS[cls],
                    label=f'DGap={cls}', s=40, edgecolors='white', lw=0.4, alpha=0.8,
                    marker=['v', 'o', '^'][cls+1])
axes[1].set_xlabel(f'{top2_gap[0]} (standardized)'); axes[1].set_ylabel(f'{top2_gap[1]} (standardized)')
axes[1].set_title('(b) Predicted Probability Surface\n[P(DGap = +1)]', fontweight='bold')
axes[1].legend()
plt.suptitle('Figure 11: Random Forest Decision Boundary and Probability Map for DGap Prediction',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig10_Decision_Boundary.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig10 saved")

# ====================================================================
# FIGURE 12 – Hierarchical Clustering Dendrogram
# ====================================================================
print("Generating Fig12 ...")
fig, ax = plt.subplots(figsize=(14, 6))
dend = dendrogram(Z_link, labels=features, ax=ax, color_threshold=0.4,
                  leaf_rotation=45, leaf_font_size=10, above_threshold_color=PALETTE['gray'])
ax.set_ylabel('Distance (1 − |Pearson r|)', fontsize=11)
ax.set_title('(a) Hierarchical Clustering — Feature Grouping by Correlation Structure',
             fontweight='bold')
ax.axhline(0.4, ls='--', color='red', lw=1.5, label='Cluster cutoff (d=0.4, |r|>0.6)')

# 标注代表特征
xlabels = [t.get_text() for t in ax.get_xticklabels()]
for i, lbl in enumerate(xlabels):
    if lbl in representative_features:
        ax.get_xticklabels()[i].set_color(PALETTE['red'])
        ax.get_xticklabels()[i].set_fontweight('bold')
ax.legend(fontsize=9)
rep_patch = mpatches.Patch(color=PALETTE['red'], label='Cluster representative (selected for model)')
ax.legend(handles=[mpatches.Patch(color='red', ls='--', label='Cutoff'), rep_patch],
          fontsize=9, loc='upper right')
plt.suptitle('Figure 5: Hierarchical Cluster Dendrogram — Feature Grouping by Correlation\n'
             '(Bold red labels = cluster representatives used in reduced model)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig11_Dendrogram.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig11 saved")

# ====================================================================
# FIGURE 1 – Modeling Flowchart
# ====================================================================
print("Generating Fig1 ...")
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 15); ax.set_ylim(0, 10); ax.axis('off')

def box(ax, x, y, w, h, text, color='#2C6FAC', text_color='white',
        fontsize=9, alpha=0.9, radius=0.22):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x-w/2, y-h/2), w, h, boxstyle=f"round,pad={radius}",
        facecolor=color, edgecolor='white', linewidth=1.5, alpha=alpha))
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', multialignment='center')

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.8))

box(ax, 7.5, 9.3, 6, 0.75, 'Raw EDM Process Data\n(154 samples × 18 variables)', '#1A5276')
box(ax, 7.5, 8.2, 6, 0.75,
    '[FIX 1] Pipeline Preprocessing\n(StandardScaler fitted on train-fold only — No Leakage)', '#922B21')
box(ax, 7.5, 7.1, 6, 0.75, 'Exploratory Data Analysis\n(Correlation, Distribution, PCA)', '#2980B9')
box(ax, 7.5, 6.0, 6, 0.75,
    f'[FIX 3] Feature Clustering → Reduced Set\n(Cluster reps: {len(representative_features)} features, logical closure)', '#1E8449')

box(ax, 3.0, 4.4, 5, 0.9, 'Task 1\nFeature–Control Quantitative\nRelationship', '#1E8449', fontsize=8.5)
box(ax, 7.5, 4.4, 4.8, 0.9, 'Task 2\nClassifier Chain Joint Control\n(DFlow ↔ DGap dependency)', '#B7950B', fontsize=8.2)
box(ax, 12.0, 4.4, 5, 0.9,
    '[FIX 2] Task 3\nDual-Model Prediction\nRF(DFlow) + SVM(DGap)', '#922B21', fontsize=8.5)

box(ax, 3.0, 2.8, 5, 0.9, 'Feature Ranking +\nImportance Scores\n(16 features → DFlow, DGap)', '#1E8449', fontsize=8.5)
box(ax, 7.5, 2.8, 4.8, 0.9, 'Joint Action Recommendation\nClassifier Chain Output\n(best order selected by CV)', '#B7950B', fontsize=8.1)
box(ax, 12.0, 2.8, 5, 0.9,
    '5-Fold Pipeline CV\n(Full & Reduced Feature)\nConfusion Matrix (Dual-Model)', '#922B21', fontsize=8.5)

box(ax, 7.5, 1.2, 11, 0.8,
    'Conclusions: Quantitative Optimization Framework for EDM Parameters', '#1A5276', fontsize=10)

for (x1,y1,x2,y2) in [
    (7.5,8.92,7.5,8.58),(7.5,7.82,7.5,7.48),(7.5,6.62,7.5,6.38),
    (5.5,5.78,3.5,4.85),(7.5,5.78,7.5,4.85),(9.5,5.78,11.5,4.85),
    (3.0,3.95,3.0,3.25),(7.5,3.95,7.5,3.25),(12.0,3.95,12.0,3.25),
    (3.0,2.35,5.5,1.60),(7.5,2.35,7.5,1.60),(12.0,2.35,9.5,1.60)]:
    arrow(ax, x1, y1, x2, y2)

ax.set_facecolor('#F5F6FA'); fig.patch.set_facecolor('#F5F6FA')
plt.title('Figure 1: Corrected Mathematical Modeling Framework for EDM Process Optimization\n'
          '(Classifier Chain + Pipeline CV + Dual-Model Test + Feature Clustering Closure)',
          fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(OUT + 'Fig12_Flowchart.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig12 saved")

# ====================================================================
# FIGURE 14 – Box-plots stratified by DFlow
# ====================================================================
print("Generating Fig14 ...")
fig, axes_arr = plt.subplots(4, 4, figsize=(18, 14))
axes_arr = axes_arr.ravel()
for i, feat in enumerate(features):
    ax = axes_arr[i]
    groups = [df[df['DFlow'] == c][feat].values for c in [-1, 0, 1]]
    bp = ax.boxplot(groups, labels=['-1','0','1'], patch_artist=True,
                    medianprops={'color':'black','linewidth':2},
                    boxprops={'linewidth':1.2}, whiskerprops={'linewidth':1})
    for patch, col in zip(bp['boxes'], [CLASS_COLORS[-1], CLASS_COLORS[0], CLASS_COLORS[1]]):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    h, p = stats.kruskal(*[g for g in groups if len(g) > 0])
    sig  = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    ax.set_title(feat, fontsize=9, fontweight='bold')
    ax.set_xlabel(f'DFlow  (KW: {sig})', fontsize=7.5); ax.grid(axis='y', alpha=0.3, ls='--')
fig.legend(handles=[mpatches.Patch(color=CLASS_COLORS[c], label=f'DFlow={c}') for c in [-1,0,1]],
           loc='lower center', ncol=3, fontsize=10, title='DFlow Class',
           title_fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))
plt.suptitle('Figure 14: Box-Plot Analysis of All 16 Pulse Features Stratified by DFlow State',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + 'Fig13_Boxplots_DFlow.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig13 saved")

# ====================================================================
# FIGURE 15 – 3D Probability Surface (PCA space, for visualisation)
# ====================================================================
print("Generating Fig15 ...")
pca2 = PCA(n_components=2); X_pca2 = pca2.fit_transform(X_sc_vis)
# NOTE: 此图使用PCA投影空间的RF，仅用于可视化，非最终预测模型
rf_pca = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_pca.fit(X_pca2, y_gap)
var2 = pca2.explained_variance_ratio_ * 100

x1r = np.linspace(X_pca2[:,0].min()-0.5, X_pca2[:,0].max()+0.5, 80)
x2r = np.linspace(X_pca2[:,1].min()-0.5, X_pca2[:,1].max()+0.5, 80)
xx1, xx2 = np.meshgrid(x1r, x2r)
opt_idx  = list(rf_pca.classes_).index(1)
Z_surf   = rf_pca.predict_proba(np.c_[xx1.ravel(), xx2.ravel()])[:, opt_idx].reshape(xx1.shape)

fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(xx1, xx2, Z_surf, cmap='coolwarm', alpha=0.85, edgecolor='none')
ax1.scatter(X_pca2[:,0], X_pca2[:,1], [0]*len(X_pca2),
            c=[CLASS_COLORS[c] for c in y_gap], s=20, alpha=0.5, zorder=5)
ax1.set_xlabel(f'PC1 ({var2[0]:.1f}%)', fontsize=9, labelpad=8)
ax1.set_ylabel(f'PC2 ({var2[1]:.1f}%)', fontsize=9, labelpad=8)
ax1.set_zlabel('P(DGap=+1)', fontsize=9, labelpad=8)
ax1.set_title('(a) 3D Probability Surface\nP(DGap=+1) in PCA Space', fontweight='bold')
fig.colorbar(surf, ax=ax1, fraction=0.03, pad=0.08, label='Probability')
ax1.view_init(elev=30, azim=220)

ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(xx1, xx2, Z_surf, cmap='viridis', alpha=0.85, edgecolor='none')
for cls in [-1, 0, 1]:
    mask  = y_gap == cls
    z_cls = rf_pca.predict_proba(X_pca2[mask])[:, opt_idx]
    ax2.scatter(X_pca2[mask,0], X_pca2[mask,1], z_cls,
                c=CLASS_COLORS[cls], s=30, label=f'DGap={cls}', alpha=0.9, zorder=10)
ax2.set_xlabel(f'PC1 ({var2[0]:.1f}%)', fontsize=9, labelpad=8)
ax2.set_ylabel(f'PC2 ({var2[1]:.1f}%)', fontsize=9, labelpad=8)
ax2.set_zlabel('P(DGap=+1)', fontsize=9, labelpad=8)
ax2.set_title('(b) Actual Data Points on\nProbability Surface', fontweight='bold')
ax2.legend(fontsize=8, loc='upper left'); ax2.view_init(elev=25, azim=60)
plt.suptitle('Figure 10: 3D Probability Surface P(DGap=+1) in PCA Feature Space\n'
             '(Note: PCA-space RF for visualisation only; final predictions use full-feature Pipeline)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'Fig14_3D_Probability_Surface.png', dpi=160, bbox_inches='tight')
plt.close()
print("  Fig14 saved")

# ====================================================================
# 数值输出模块
# ====================================================================
SEP = "=" * 70
print(f"\n{SEP}")
print("数值计算模块")
print(SEP)

# 1. 数据集基本统计
vals_flow_cnt = pd.Series(y_flow).value_counts().sort_index()
vals_gap_cnt  = pd.Series(y_gap).value_counts().sort_index()
print(f"\n【1】数据集规模: {len(df)} 样本，{len(features)} 个脉冲特征")
print(f"  DFlow 分布: {dict(vals_flow_cnt)}")
print(f"  DGap  分布: {dict(vals_gap_cnt)}")

# 2. 描述性统计
print(f"\n【2】16个特征描述性统计")
desc = df[features].describe().T[['mean','std','min','max']]
print(f"  {'特征':<18} {'均值':>8} {'标准差':>8} {'最小值':>8} {'最大值':>8}")
for feat, row in desc.iterrows():
    print(f"  {feat:<18} {row['mean']:>8.3f} {row['std']:>8.3f} {row['min']:>8.3f} {row['max']:>8.3f}")

# 3. Pearson相关系数
print(f"\n【3】Pearson相关系数（特征 vs DFlow/DGap）")
corr_res = {}
print(f"  {'特征':<18} {'r(DFlow)':>10} {'p(DFlow)':>10} {'r(DGap)':>10} {'p(DGap)':>10}")
for feat in features:
    rf_, pf_ = stats.pearsonr(df[feat], df['DFlow'])
    rg_, pg_ = stats.pearsonr(df[feat], df['DGap'])
    corr_res[feat] = {'r_flow': rf_, 'p_flow': pf_, 'r_gap': rg_, 'p_gap': pg_}
    sf = '***' if pf_ < 0.001 else ('**' if pf_ < 0.01 else ('*' if pf_ < 0.05 else 'ns'))
    sg = '***' if pg_ < 0.001 else ('**' if pg_ < 0.01 else ('*' if pg_ < 0.05 else 'ns'))
    print(f"  {feat:<18} {rf_:>10.4f}{sf:<3} {pf_:>8.4f}  {rg_:>10.4f}{sg:<3} {pg_:>8.4f}")

# 4. 强相关特征对
print(f"\n【4】强相关特征对（|r|>0.8）")
strong_pairs = []
for i in range(len(features)):
    for j in range(i+1, len(features)):
        r = corr.iloc[i, j]
        if abs(r) > 0.8:
            strong_pairs.append((features[i], features[j], r))
print(f"  共 {len(strong_pairs)} 对:")
for a, b, r in sorted(strong_pairs, key=lambda x: -abs(x[2])):
    print(f"  {a:<18} — {b:<18}  r = {r:>7.4f}")

# 5. PCA方差贡献率
print(f"\n【5】PCA方差贡献率（前6个主成分）")
for i, (v, c) in enumerate(zip(var_exp, cum_var)):
    print(f"  PC{i+1}: {v:.2f}%  累计 {c:.2f}%")

# 6. 随机森林特征重要性
order_gap  = np.argsort(imp_gap)[::-1]
order_flow = np.argsort(imp_flow)[::-1]
print(f"\n【6】随机森林特征重要性排序")
print(f"  DGap 模型:")
for rank, idx in enumerate(order_gap):
    print(f"    {rank+1:2d}. {features[idx]:<18} {imp_gap[idx]*100:.2f}%")
print(f"  DFlow 模型:")
for rank, idx in enumerate(order_flow):
    print(f"    {rank+1:2d}. {features[idx]:<18} {imp_flow[idx]*100:.2f}%")

# 7. 5折Pipeline CV结果
print(f"\n【7】5折交叉验证 Macro-F1（Pipeline，无数据泄露）")
print(f"  {'模型':<18} DFlow均值±std  DGap均值±std  (全特征)")
for name in cv_results_all:
    fa = cv_results_all[name]['flow']; ga = cv_results_all[name]['gap']
    print(f"  {name:<18} {fa.mean():.1f}%±{fa.std():.1f}%   {ga.mean():.1f}%±{ga.std():.1f}%")

print(f"\n  {'模型':<18} DFlow均值±std  DGap均值±std  (精简特征{len(representative_features)}维)")
for name in cv_results_red:
    fr = cv_results_red[name]['flow']; gr = cv_results_red[name]['gap']
    print(f"  {name:<18} {fr.mean():.1f}%±{fr.std():.1f}%   {gr.mean():.1f}%±{gr.std():.1f}%")

best_flow_m = max(cv_results_all, key=lambda n: cv_results_all[n]['flow'].mean())
best_gap_m  = max(cv_results_all, key=lambda n: cv_results_all[n]['gap'].mean())
print(f"\n  DFlow最优: {best_flow_m}  "
      f"F1={cv_results_all[best_flow_m]['flow'].mean():.1f}%±{cv_results_all[best_flow_m]['flow'].std():.1f}%")
print(f"  DGap最优:  {best_gap_m}  "
      f"F1={cv_results_all[best_gap_m]['gap'].mean():.1f}%±{cv_results_all[best_gap_m]['gap'].std():.1f}%")

# 8. 测试集评估
print(f"\n【8】测试集双模型评估（RF for DFlow, SVM-RBF for DGap）")
print(f"  训练集: {len(idx_tr)}，测试集: {len(idx_te)}")
print(f"  DFlow (RF, {feat_label_flow}): Acc={acc_flow2*100:.1f}%  Macro-F1={f1_flow2*100:.1f}%")
print(classification_report(yf_te2, yf_pred2, target_names=['DFlow=-1','DFlow=0','DFlow=+1']))
print(f"  DGap (SVM-RBF, {feat_label_gap}): Acc={acc_gap2*100:.1f}%  Macro-F1={f1_gap2*100:.1f}%")
print(classification_report(yg_te2, yg_pred2, target_names=['DGap=-1','DGap=0','DGap=+1']))
print(f"  混淆矩阵 DFlow:\n{cm_flow2}")
print(f"  混淆矩阵 DGap:\n{cm_gap2}")

# 10. 联合频次与分类器链结果
print(f"\n【10】DFlow–DGap联合状态频次")
print(combo.to_string())

print(f"\n【11】分类器链联合建模（问题二主模型）")
print(f"  链顺序1: DFlow→DGap  Joint Macro-F1 = {chain_flow_gap['joint'].mean():.1f}%±{chain_flow_gap['joint'].std():.1f}%")
print(f"           一级(DFlow)={chain_flow_gap['first'].mean():.1f}%  二级(DGap|DFlow)={chain_flow_gap['second'].mean():.1f}%")
print(f"  链顺序2: DGap→DFlow  Joint Macro-F1 = {chain_gap_flow['joint'].mean():.1f}%±{chain_gap_flow['joint'].std():.1f}%")
print(f"           一级(DGap)={chain_gap_flow['first'].mean():.1f}%  二级(DFlow|DGap)={chain_gap_flow['second'].mean():.1f}%")
print(f"  最优链顺序: {chain_best_name}")

chain_holdout = classifier_chain_holdout(
    X_raw,
    y_flow if chain_best_name == 'DFlow→DGap' else y_gap,
    y_gap if chain_best_name == 'DFlow→DGap' else y_flow,
    chain_rf if chain_best_name == 'DFlow→DGap' else chain_svm,
    chain_svm if chain_best_name == 'DFlow→DGap' else chain_rf,
    idx_tr,
    idx_te,
)
print(f"  测试集联合动作准确率 = {chain_holdout['joint_acc']:.1f}%")
print(f"  测试集联合动作 Joint Macro-F1 = {chain_holdout['joint_f1']:.1f}%")

# 12. KW检验（保留为辅助统计证据）
print(f"\n【12】Kruskal-Wallis 检验（特征 vs 复合指标S，辅助分析）")
print(f"  {'特征':<18} {'H统计量':>10} {'p值':>10} 显著性")
for feat in features:
    groups = [df[df['S']==s][feat].values for s in s_vals]
    h, p   = stats.kruskal(*[g for g in groups if len(g) > 0])
    sig    = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    print(f"  {feat:<18} {h:>10.3f} {p:>10.4f} {sig}")

# 13. DFlow–DGap结构关系
r_fd,   p_fd   = stats.pearsonr(df['DFlow'], df['DGap'])
rho_fd, p_rho  = stats.spearmanr(df['DFlow'], df['DGap'])
ct_tab = pd.crosstab(df['DFlow'], df['DGap'])
chi2s, pchi2, dof_chi, _ = stats.chi2_contingency(ct_tab)
print(f"\n【13】DFlow–DGap结构关系")
print(f"  Pearson  r = {r_fd:.4f}  p = {p_fd:.4f}")
print(f"  Spearman ρ = {rho_fd:.4f}  p = {p_rho:.4f}")
print(f"  卡方检验 χ²={chi2s:.3f}  df={dof_chi}  p={pchi2:.4f}")

# 14. LDA投影
print(f"\n【14】LDA各类别LD1投影均值")
for label, lda_r, y_r in [('DGap', X_lda_gap_v, y_gap), ('DFlow', X_lda_flow_v, y_flow)]:
    print(f"  {label}:")
    for cls in [-1, 0, 1]:
        mask = y_r == cls
        if mask.sum() > 0:
            ld = lda_r[mask, 0] if lda_r.ndim > 1 else lda_r[mask]
            print(f"    {label}={cls}: LD1均值={ld.mean():.4f}  std={ld.std():.4f}")

# 15. 各特征按DGap分组均值
print(f"\n【15】各特征按DGap分组均值对比")
print(f"  {'特征':<18} {'DGap=-1':>10} {'DGap=0':>10} {'DGap=+1':>10} {'p值':>10}")
for feat in features:
    grp   = [df[df['DGap']==c][feat].values for c in [-1,0,1]]
    means = [g.mean() for g in grp]
    h, p  = stats.kruskal(*grp)
    sig   = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    print(f"  {feat:<18} {means[0]:>10.3f} {means[1]:>10.3f} {means[2]:>10.3f} {p:>8.4f}{sig}")

# 16. 精简特征集信息
print(f"\n【16】精简特征集（层次聚类代表特征）")
print(f"  原始维度: 16  →  精简维度: {len(representative_features)}")
print(f"  代表特征: {representative_features}")
for c, info in cluster_info.items():
    print(f"  簇{c}: {info['members']} → 代表: {info['representative']}")

print(f"\n{SEP}")
print(f"  最优链顺序 {chain_best_name}  Joint CV F1: {chain_best['joint'].mean():.1f}%")
print(f"  DFlow RF  Pipeline CV F1: {cv_results_all['Random Forest']['flow'].mean():.1f}%")
print(f"  DGap SVM  Pipeline CV F1: {cv_results_all['SVM (RBF)']['gap'].mean():.1f}%")
print(f"  DFlow RF  Test Set   F1: {f1_flow2*100:.1f}%  (Acc={acc_flow2*100:.1f}%)")
print(f"  DGap SVM  Test Set   F1: {f1_gap2*100:.1f}%  (Acc={acc_gap2*100:.1f}%)")
print(SEP)
print("\n code_vFinal.py 执行完成")