import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# 全局字体设置
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX 字体接近 Times New Roman
# 或者使用 'stixsans' 等
plt.rcParams['font.size'] = 16         # 全局默认字体大小（可选项）

th1 = np.array([0.6, 0.7, 0.8, 0.9])
th2 = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
acc = np.array([[82.0, 83.3, 82.0, 83.3, 82.0],
                [80.8, 84.5, 83.3, 82.0, 81.6],
                [81.2, 81.6, 82.4, 82.4, 80.4],
                [75.9, 83.3, 81.6, 80.0, 80.4]])

X, Y = np.meshgrid(th2, th1)
points = np.column_stack([X.ravel(), Y.ravel()])
values = acc.ravel()

xi = np.linspace(th2.min(), th2.max(), 200)
yi = np.linspace(th1.min(), th1.max(), 200)
Xi, Yi = np.meshgrid(xi, yi)
Zi = griddata(points, values, (Xi, Yi), method='cubic')

fig, ax = plt.subplots(figsize=(8, 6))
cf = ax.contourf(Xi, Yi, Zi, levels=20, cmap='viridis', vmin=75, vmax=85)
ct = ax.contour(Xi, Yi, Zi, levels=8, colors='black', linewidths=0.5)
ax.clabel(ct, inline=True, fontsize=16, fmt='%.1f')
ax.scatter(X.ravel(), Y.ravel(), color='black', s=20, zorder=5, clip_on=False)

ax.set_xlim(th2.min(), th2.max())
ax.set_ylim(th1.min(), th1.max())

# ---- 字体大小设置 ----
# 轴标题
ax.set_xlabel('$th_2$', fontsize=20)
ax.set_ylabel('$th_1$', fontsize=20)
# ax.set_title('Effect of Thresholds on Model Accuracy', fontsize=20, pad=15)

# 刻度标签
for label in ax.get_xticklabels() + ax.get_yticklabels():
    # label.set_fontweight('bold')
    label.set_fontsize(16)   

# Colorbar
cbar = plt.colorbar(cf, ax=ax, label='Accuracy (%)')
cbar.ax.tick_params(labelsize=16)          # colorbar 刻度字体
cbar.set_label('Accuracy (%)', fontsize=20) # colorbar 标题
# --------------------

ax.set_xticks(th2)
ax.set_yticks(th1)
plt.tight_layout()
plt.savefig('thresholds.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()