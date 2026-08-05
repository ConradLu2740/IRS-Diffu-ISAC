import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

base_dir = r'c:\trae_solo\workspace\IRS_Diffu_ISAC\results_complete'

# 加载点云数据
def load_pc(path):
    return np.load(path).astype(np.float32)

# 三个场景的数据路径
scenes = {
    'IRS+随机相位': {
        'gt': os.path.join(base_dir, 'outputs_npy', 'PC_gt_{:04d}.npy'),
        'pred': os.path.join(base_dir, 'outputs_npy', 'PC_hat_{:04d}.npy'),
    },
    '无IRS': {
        'gt': os.path.join(base_dir, 'outputs_ldm_no_irs', 'result', 'sample_{:04d}_gt.npy'),
        'pred': os.path.join(base_dir, 'outputs_ldm_no_irs', 'result', 'sample_{:04d}_pred.npy'),
    },
    '随机IRS': {
        'gt': os.path.join(base_dir, 'outputs_ldm_random_irs', 'result', 'sample_{:04d}_gt.npy'),
        'pred': os.path.join(base_dir, 'outputs_ldm_random_irs', 'result', 'sample_{:04d}_pred.npy'),
    }
}

# 计算Chamfer Distance
def chamfer_distance(pc1, pc2):
    # pc1, pc2: (N, 3)
    dist_matrix = np.sqrt(((pc1[:, None, :] - pc2[None, :, :]) ** 2).sum(axis=2))
    dist1 = dist_matrix.min(axis=1).mean()
    dist2 = dist_matrix.min(axis=0).mean()
    return dist1 + dist2

# 收集样本的CD数据 (只取sample_0001)
cd_data = {name: [] for name in scenes}
sample_ids = [1]

for sid in sample_ids:
    for scene_name, paths in scenes.items():
        gt = load_pc(paths['gt'].format(sid))
        pred = load_pc(paths['pred'].format(sid))
        cd = chamfer_distance(gt, pred)
        cd_data[scene_name].append(cd)
        print(f"[{scene_name}] sample_{sid:04d} CD: {cd:.6f}")

# 创建大图
fig = plt.figure(figsize=(24, 16))

# 1. Chamfer Distance 柱状图
ax1 = fig.add_subplot(2, 3, 1)
x = np.arange(len(sample_ids))
width = 0.25
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, (scene_name, cds) in enumerate(cd_data.items()):
    ax1.bar(x + i*width, cds, width, label=scene_name, color=colors[i])
ax1.set_xlabel('Sample ID')
ax1.set_ylabel('Chamfer Distance')
ax1.set_title('Chamfer Distance Comparison (Lower is Better)')
ax1.set_xticks(x + width)
ax1.set_xticklabels([f'{sid:04d}' for sid in sample_ids])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. 平均CD对比
ax2 = fig.add_subplot(2, 3, 2)
avg_cds = [np.mean(cd_data[name]) for name in scenes]
bars = ax2.bar(list(scenes.keys()), avg_cds, color=colors)
ax2.set_ylabel('Average Chamfer Distance')
ax2.set_title('Average CD Comparison')
for bar, val in zip(bars, avg_cds):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
             f'{val:.6f}', ha='center', va='bottom', fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# 3-8. 点云可视化 (sample_0001)
sample_id = 1
for idx, (scene_name, paths) in enumerate(scenes.items()):
    gt = load_pc(paths['gt'].format(sample_id))
    pred = load_pc(paths['pred'].format(sample_id))
    cd = cd_data[scene_name][0]
    
    # GT
    ax_gt = fig.add_subplot(2, 3, idx+3, projection='3d')
    ax_gt.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='blue', s=1, alpha=0.6)
    ax_gt.set_title(f'{scene_name}\nGT (Sample {sample_id:04d})')
    ax_gt.set_xlabel('X')
    ax_gt.set_ylabel('Y')
    ax_gt.set_zlabel('Z')
    
# 创建第二个图：Pred对比
fig2 = plt.figure(figsize=(24, 8))
for idx, (scene_name, paths) in enumerate(scenes.items()):
    pred = load_pc(paths['pred'].format(sample_id))
    cd = cd_data[scene_name][0]
    
    ax_pred = fig2.add_subplot(1, 3, idx+1, projection='3d')
    ax_pred.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='red', s=1, alpha=0.6)
    ax_pred.set_title(f'{scene_name}\nPred (CD: {cd:.6f})')
    ax_pred.set_xlabel('X')
    ax_pred.set_ylabel('Y')
    ax_pred.set_zlabel('Z')

plt.tight_layout()
fig.savefig(r'c:\trae_solo\workspace\IRS_Diffu_ISAC\comparison_results.png', dpi=150, bbox_inches='tight')
fig2.savefig(r'c:\trae_solo\workspace\IRS_Diffu_ISAC\comparison_pred.png', dpi=150, bbox_inches='tight')
print("\n图表已保存:")
print("  - comparison_results.png (综合对比)")
print("  - comparison_pred.png (预测点云对比)")

# 打印汇总
print("\n" + "="*60)
print("汇总对比")
print("="*60)
for scene_name in scenes:
    cds = cd_data[scene_name]
    print(f"{scene_name:12s}: 平均CD = {np.mean(cds):.6f}, 最小 = {np.min(cds):.6f}, 最大 = {np.max(cds):.6f}")
