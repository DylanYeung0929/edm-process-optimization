import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import numpy as np

# ==========================================
# 1. 字体与风格配置
# ==========================================
plt.rcParams.update({
    'font.family': 'serif',
    # 锁定英文字体为 DejaVu Serif
    'font.serif': ['DejaVu Serif', 'Bitstream Vera Serif', 'Times New Roman'],
    'mathtext.fontset': 'dejavuserif',
    'axes.formatter.use_mathtext': True,
    'axes.unicode_minus': False
})



def get_bold_chn_font(fontsize=15):
    for name in ['Heiti TC', 'STHeiti', 'Hei', 'SimHei', 'Arial Unicode MS']:
        fp = FontProperties(family=name, size=fontsize, weight='bold')
        return fp
    return FontProperties(size=fontsize, weight='bold')


chn_font_main = get_bold_chn_font(fontsize=16)
chn_font_sub = get_bold_chn_font(fontsize=17)

print("Generating Flowchart with ENHANCED Bold Heiti TC and Safe Brackets...")


fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis('off')


# ==========================================
# 2. 定义带阴影和双字体渲染的方框函数
# ==========================================
def box(ax, x, y, w, h, text, facecolor='#2C6FAC', text_color='white'):
    # 1. 绘制阴影
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - w / 2 + 0.06, y - h / 2 - 0.06), w, h, boxstyle="round,pad=0.22",
        facecolor='black', edgecolor='none', alpha=0.12))

    # 2. 绘制主方框
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.22",
        facecolor=facecolor, edgecolor='white', linewidth=1.8, alpha=0.98))

    lines = text.split('\n')
    header = lines[0]
    body = "\n".join(lines[1:]) if len(lines) > 1 else ""

    if body:
        ax.text(x, y + 0.20, header, ha='center', va='center',
                fontfamily='DejaVu Serif', fontweight='bold', fontsize=16, color=text_color, zorder=5)
        ax.text(x, y - 0.20, body, ha='center', va='center',
                fontproperties=chn_font_sub, color=text_color, linespacing=1.6, zorder=5)
    else:
        ax.text(x, y, header, ha='center', va='center',
                fontfamily='DejaVu Serif', fontweight='bold', fontsize=16, color=text_color, zorder=5)


# ==========================================
# 3. 定义避让箭头函数
# ==========================================
def draw_safe_arrow(ax, p1, p2, rad=0.1):
    (x1, y1), (x2, y2) = p1, p2
    gap = 0.22
    if x1 == x2:
        if y1 > y2:
            ax.annotate('', xy=(x2, y2 + gap), xytext=(x1, y1 - gap),
                        arrowprops=dict(arrowstyle='->', color='#555555', lw=2.0, connectionstyle="arc3,rad=0"))
        return
    vx, vy = x2 - x1, y2 - y1
    length = np.sqrt(vx ** 2 + vy ** 2)
    ux, uy = vx / length * gap, vy / length * gap
    ax.annotate('', xy=(x2 - ux, y2 - uy), xytext=(x1 + ux, y1 + uy),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=2.0, connectionstyle=f"arc3,rad={rad}"))


# ==========================================
# 4. 布局与内容定义
# ==========================================
Y_STEPS = [10.1, 8.8, 7.5, 6.2]
box(ax, 9.0, Y_STEPS[0], 7.8, 0.6, 'Raw EDM Process Data\n原始电火花加工数据集 (154组样本)', '#1A5276')
box(ax, 9.0, Y_STEPS[1], 7.8, 0.6, 'Pipeline Data Preprocessing\nPipeline 数据预处理 (彻底杜绝泄露)', '#922B21')
box(ax, 9.0, Y_STEPS[2], 7.8, 0.6, 'Exploratory Data Analysis (EDA)\n探索性分析 (相关性、分布及 PCA)', '#2980B9')
box(ax, 9.0, Y_STEPS[3], 7.8, 0.6, 'Feature Engineering: Clustering\n特征工程：精简提取 13 维特征', '#1E8449')

Y_BRANCH = 4.4
TX = [3.2, 9.0, 14.8]
box(ax, TX[0], Y_BRANCH, 5.4, 1.2, '[Task 1]\n特征与控制变量量化关系识别', '#1E8449')
box(ax, TX[1], Y_BRANCH, 5.4, 1.2, '[Task 2]\n分类器链协同调控模型', '#B7950B')
box(ax, TX[2], Y_BRANCH, 5.4, 1.2, '[Task 3]\n最优参数双模型预测策略', '#922B21')

Y_MERGE = 2.4
box(ax, TX[0], Y_MERGE, 5.4, 1.0, 'Task 1 Results\n特征重要性结论与脉冲物理机理', '#1E8449')
box(ax, TX[1], Y_MERGE, 5.4, 1.0, 'Task 2 Results\n协同动作推荐与最佳调节顺序', '#B7950B')
box(ax, TX[2], Y_MERGE, 5.4, 1.0, 'Task 3 Evaluation\n分层 5 折交叉验证与混淆矩阵', '#922B21')

box(ax, 9.0, 0.7, 15.0, 0.6, 'Final Conclusion: Quantitative Modeling Framework for EDM Optimization', '#1A5276')

# ==========================================
# 5. 连线与标题
# ==========================================
draw_safe_arrow(ax, (9.0, 10.1), (9.0, 8.8))
draw_safe_arrow(ax, (9.0, 8.8), (9.0, 7.5))
draw_safe_arrow(ax, (9.0, 7.5), (9.0, 6.2))

Y_FC_BOT = 6.2 - 0.3
draw_safe_arrow(ax, (9.0, Y_FC_BOT), (TX[0], Y_BRANCH + 0.60), rad=-0.05)
draw_safe_arrow(ax, (9.0, Y_FC_BOT), (TX[1], Y_BRANCH + 0.60), rad=0)
draw_safe_arrow(ax, (9.0, Y_FC_BOT), (TX[2], Y_BRANCH + 0.60), rad=0.05)

for tx in TX: draw_safe_arrow(ax, (tx, Y_BRANCH - 0.60), (tx, Y_MERGE + 0.5))

Y_M_BOT = Y_MERGE - 0.5
draw_safe_arrow(ax, (TX[0], Y_M_BOT), (9.0, 1.0), rad=-0.12)
draw_safe_arrow(ax, (TX[1], Y_M_BOT), (9.0, 1.0), rad=0)
draw_safe_arrow(ax, (TX[2], Y_M_BOT), (9.0, 1.0), rad=0.12)

plt.title('Figure 1: Integrated Modeling Lifecycle for EDM Process Parameter Optimization\n'
          '(EDA -> Classifier Chain -> Pipeline-based Dual Model Strategy)',
          fontsize=18, fontweight='bold', fontfamily='DejaVu Serif', pad=40, color='#2C3E50')

plt.tight_layout(pad=2.0)
out_file = 'outputs/Fig12_Flowchart.png'
plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Success: '{out_file}' generated. Brackets fixed, Chinese text enlarged.")