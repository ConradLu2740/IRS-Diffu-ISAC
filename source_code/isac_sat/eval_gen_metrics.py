"""
eval_gen_metrics.py — 分布型生成指标：Coverage@τ / 1-NNA / MMD-CD

背景（外部评审 W5）：CD / F-Score / Voxel IoU 只度量**逐样本重建误差**——
对离群点与密度不敏感，也无法回答"生成分布是否覆盖参考分布"与
"是否记忆了参考样本"。本脚本在同一归一化点云单位下补齐三个分布型指标：

  - Coverage@τ : 参考点云中，至少有一个生成样本落在 CD < τ 内的比例。
                 τ 与 F-Score 阈值一致（0.1 / 0.2）。低覆盖率 = 模式丢失。
  - 1-NNA      : 生成点云在"生成 ∪ 参考"混合池中的最近邻落在生成侧的比例。
                 ≈0.5 = 与参考不可区分；显著 >0.5 = 记忆化 / 模式坍塌信号。
  - MMD-CD     : 以 CD 为核的 MMD 估计：mean CD(gen→ref) 与 mean CD(ref→gen)
                 双向成对距离（CD 单位）。另报参考集自 MMD-CD（i≠j）作为
                 不可约下界，使读数可校准。

协议（与 verify_fm_distill_diversity.py / train_fm_distill.py 的 C1 HRRP
评估协议一致；**不重训、不改 checkpoint**）：
  - C1 HRRP 条件模型（sat_model_c1/sat）：
      VAE / DDPM(NFE=T=100) / FM-OT-CFM(NFE=1, 10) / VAE oracle
  - 蒸馏 1 步学生（sat_model_distill/student_1step.pth），与 teacher 共享
    condenc；x0 用种子 999 的独立生成器（复现蒸馏评估约定）
  - 同一测试批：--eval_seed 重播种后构建（与蒸馏/多样性评估同一批）
  - 样本集规模 = 参考集规模（每个参考条件抽 1 个样本）
  - CD 全部复用 train.chamfer_distance_loss（不重造 CD 实现）

用法：
  python eval_gen_metrics.py --ckpt_dir ./sat_model_c1 \
      --student_dir ./sat_model_distill --save_dir ./sat_model_cmp
"""

import os
import json
import argparse
import random
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_LEGACY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "legacy")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

import setup_sat as ss
from data_sat import SatROIDataset, SatScenarioChannels
from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn
from train import DDPMScheduler, sample_conditional_1D, chamfer_distance_loss
from fm_utils import T_EMB_SCALE, sample_conditional_FM
from compare_gen import _PairView
from eval_sat import f_score


# ----------------------------------------------------------------------
# 数据批 + checkpoint 加载（协议同 C1 HRRP 系列脚本）
# ----------------------------------------------------------------------

def build_eval_batch(args, device):
    """C1 HRRP 测试批：--eval_seed 重播种后构建（协议同 verify_fm_distill_diversity）。"""
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed)
    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode="sat", device=device)
    test_ds = SatROIDataset(args.test_data, channels, num_points=args.num_points,
                            device=device, tau=args.tau, phase_mode=args.phase_mode,
                            cond_feat=args.cond_feat)
    test_loader = DataLoader(_PairView(test_ds), batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    pc_gt, cond = next(iter(test_loader))
    return pc_gt[:args.n_eval].to(device), cond[:args.n_eval].to(device)


def load_c1_models(args, device):
    """C1 HRRP checkpoint：VAE + DDPM + FM teacher（*_best.pth）。"""
    src = os.path.join(args.ckpt_dir, "sat")
    for name in ("vae_best.pth", "condenc_best.pth", "epsnet_best.pth",
                 "condenc_fm_best.pth", "vnet_fm_best.pth", "latent_stats.pth"):
        path = os.path.join(src, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少 checkpoint: {path}（不重训，直接上报缺失）")

    ce_sd = torch.load(os.path.join(src, "condenc_fm_best.pth"), map_location=device)
    cond_dim = int(ce_sd["lstm.weight_ih_l0"].shape[1])

    vae = PointVAE(num_points=args.num_points, z_dim=256).to(device)
    vae.load_state_dict(torch.load(os.path.join(src, "vae_best.pth"), map_location=device))
    condenc_d = AdvancedCondEncoder(seq_len=args.tau, input_size=cond_dim,
                                    hidden_size=128, out_emb=256).to(device)
    condenc_d.load_state_dict(torch.load(os.path.join(src, "condenc_best.pth"),
                                         map_location=device))
    epsnet = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                   depth=args.depth, num_heads=8).to(device)
    epsnet.load_state_dict(torch.load(os.path.join(src, "epsnet_best.pth"),
                                      map_location=device))
    condenc_f = AdvancedCondEncoder(seq_len=args.tau, input_size=cond_dim,
                                    hidden_size=128, out_emb=256).to(device)
    condenc_f.load_state_dict(torch.load(os.path.join(src, "condenc_fm_best.pth"),
                                         map_location=device))
    vnet = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                 depth=args.depth, num_heads=8).to(device)
    vnet.load_state_dict(torch.load(os.path.join(src, "vnet_fm_best.pth"),
                                    map_location=device))
    stats = torch.load(os.path.join(src, "latent_stats.pth"), map_location=device)
    for m in (vae, condenc_d, epsnet, condenc_f, vnet):
        m.eval()
    return {"vae": vae, "condenc_d": condenc_d, "epsnet": epsnet,
            "condenc_f": condenc_f, "vnet": vnet,
            "z_mean": stats["z_mean"], "z_std": stats["z_std"],
            "cond_dim": cond_dim, "src": src}


def load_student(args, device):
    """蒸馏 1 步学生（独立参数 vnet 同构结构；condenc 与 teacher 共享）。"""
    path = os.path.join(args.student_dir, "student_1step.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少学生 checkpoint: {path}（不重训，直接上报缺失）")
    student = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                    depth=args.depth, num_heads=8).to(device)
    student.load_state_dict(torch.load(path, map_location=device))
    student.eval()
    return student, path


# ----------------------------------------------------------------------
# 指标实现（CD 复用 train.chamfer_distance_loss）
# ----------------------------------------------------------------------

def cd_matrix(A, B):
    """逐对 CD 矩阵 [len(A), len(B)]，逐对调用仓库既有 chamfer_distance_loss。"""
    A = A.detach()
    B = B.detach()
    M = np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    for i in range(A.shape[0]):
        ai = A[i:i + 1]
        for j in range(B.shape[0]):
            M[i, j] = chamfer_distance_loss(ai, B[j:j + 1]).item()
    return M


def coverage(ref, gen, cd_tau):
    """参考点云中至少有一个生成样本 CD < τ 的比例（τ 与 F-Score 阈值一致）。"""
    D = cd_matrix(ref, gen)                       # [n_ref, n_gen]
    return float((D.min(axis=1) < cd_tau).mean())


def nna_accuracy(gen, ref):
    """生成侧 1-NNA：生成点云的最近邻（混合池内，排除自身）是生成侧的比例。

    ≈0.5 = 生成分布与参考分布不可区分；>0.5 = 记忆化/模式坍塌信号。
    """
    Dgg = cd_matrix(gen, gen)                     # [n_gen, n_gen]（对角线为自身）
    Dgr = cd_matrix(gen, ref)                     # [n_gen, n_ref]
    hits = 0.0
    for i in range(gen.shape[0]):
        d_gen = np.concatenate([Dgg[i, :i], Dgg[i, i + 1:]])
        hits += float(d_gen.min() < Dgr[i].min())
    return hits / gen.shape[0]


def nna_accuracy_ref_side(gen, ref):
    """参考侧对称版本（sanity check：理想采样器两侧都应 ≈0.5）。"""
    Drg = cd_matrix(ref, gen)                     # [n_ref, n_gen]
    Drr = cd_matrix(ref, ref)                     # [n_ref, n_ref]
    hits = 0.0
    for i in range(ref.shape[0]):
        d_ref = np.concatenate([Drr[i, :i], Drr[i, i + 1:]])
        hits += float(d_ref.min() < Drg[i].min())
    return hits / ref.shape[0]


def mmd_cd(gen, ref):
    """以 CD 为核的 MMD 估计（双向成对平均，CD 单位）。

    条件生成下 gen_i 与 ref_i 由同一条件产生（配对），i=j 项被配对关系拉低，
    故另报 offdiag（i≠j）版本——纯分布距离，剔除配对角。
    """
    Dgr = cd_matrix(gen, ref)
    Drg = cd_matrix(ref, gen)
    g2r = float(Dgr.mean())
    r2g = float(Drg.mean())
    off = Dgr[~np.eye(Dgr.shape[0], dtype=bool)]
    return {"gen_to_ref": g2r, "ref_to_gen": r2g,
            "mmd_cd": float(0.5 * (g2r + r2g)),
            "mmd_cd_offdiag": float(off.mean())}


def ref_self_mmd_cd(ref):
    """参考集自 MMD-CD（i≠j 有序对均值）：同分布两次独立抽样的不可约下界。"""
    D = cd_matrix(ref, ref)
    off = D[~np.eye(D.shape[0], dtype=bool)]
    return float(off.mean())


def pairwise_cd_internal(pcs):
    """集合内部 i≠j 成对 CD 均值（1-NNA 读数的解释量：内部越紧，1-NNA 越易触顶）。"""
    D = cd_matrix(pcs, pcs)
    off = D[~np.eye(D.shape[0], dtype=bool)]
    return float(off.mean())


# ----------------------------------------------------------------------
# 采样器（全部走仓库既有实现；学生复现 train_fm_distill 的评估约定）
# ----------------------------------------------------------------------

@torch.no_grad()
def sample_ddpm(models, cond, args, device):
    torch.manual_seed(args.sample_seed)
    return sample_conditional_1D(models["vae"], models["condenc_d"], models["epsnet"],
                                 DDPMScheduler(T=args.T, device=device), cond,
                                 models["z_mean"], models["z_std"],
                                 device=device, cfg_scale=args.cfg_scale)


@torch.no_grad()
def sample_fm(models, cond, args, device, nfe, solver="euler"):
    torch.manual_seed(args.sample_seed)
    return sample_conditional_FM(models["vae"], models["condenc_f"], models["vnet"],
                                 cond, models["z_mean"], models["z_std"],
                                 device=device, cfg_scale=args.cfg_scale,
                                 nfe=nfe, solver=solver)


@torch.no_grad()
def sample_student(models, student, cond, args, device):
    """1 步学生：z = x0 + v_s(x0, 0, c)（CFG w=2），x0 用种子 999 的独立生成器。"""
    vae, condenc = models["vae"], models["condenc_f"]
    B = cond.size(0)
    g = torch.Generator(device=device).manual_seed(args.x0_seed)
    x0 = torch.randn((B, 256), device=device, generator=g)
    c = condenc(cond)
    c_null = condenc(torch.zeros_like(cond))
    t = torch.zeros(B, device=device)
    v = student(x0, t, c_null) + args.cfg_scale * (student(x0, t, c) - student(x0, t, c_null))
    z0 = (x0 + v) * models["z_std"] + models["z_mean"]
    pc, _ = vae.decode(z0)
    return pc


@torch.no_grad()
def sample_vae_oracle(models, pc_gt):
    """VAE 上界：GT 潜变量后验均值解码（重建天花板，非采样器）。"""
    mu, _ = models["vae"].encode(pc_gt)
    pc, _ = models["vae"].decode(mu)
    return pc


# ----------------------------------------------------------------------
# 指标汇总
# ----------------------------------------------------------------------

def gen_metrics(pc_gt, pc_gen, args):
    """单个模型的分布型指标 + 逐样本 CD/F-Score（用于与表 I–II 对齐）。"""
    out = {}
    for tau in args.cd_taus:
        out[f"coverage_{tau:g}"] = coverage(pc_gt, pc_gen, tau)
    out["nna_accuracy"] = nna_accuracy(pc_gen, pc_gt)
    out["nna_accuracy_ref_side"] = nna_accuracy_ref_side(pc_gen, pc_gt)
    out.update(mmd_cd(pc_gen, pc_gt))
    out["pairwise_cd_gen_internal"] = pairwise_cd_internal(pc_gen)
    out["cd_mean"] = float(np.mean([chamfer_distance_loss(pc_gt[i:i + 1],
                                                          pc_gen[i:i + 1]).item()
                                    for i in range(pc_gt.shape[0])]))
    out["fs_0.1"] = float(np.mean([f_score(pc_gt[i], pc_gen[i], 0.1)
                                   for i in range(pc_gt.shape[0])]))
    out["fs_0.2"] = float(np.mean([f_score(pc_gt[i], pc_gen[i], 0.2)
                                   for i in range(pc_gt.shape[0])]))
    return out


def main(args):
    device = args.device
    print(f"Device: {device}")
    pc_gt, cond = build_eval_batch(args, device)
    print(f"参考集: {tuple(pc_gt.shape)}  cond: {tuple(cond.shape)}"
          f"（eval_seed={args.eval_seed}, test_data={args.test_data}, n_eval={args.n_eval}）")

    models = load_c1_models(args, device)
    print(f"C1 HRRP checkpoint 加载自 {models['src']} (cond_dim={models['cond_dim']})")

    # 参考集自身统计（读数校准的下界）
    ref_stats = {
        "n": int(pc_gt.shape[0]),
        "mmd_cd_ref_self": ref_self_mmd_cd(pc_gt),
    }
    print(f"参考集自 MMD-CD (i≠j) = {ref_stats['mmd_cd_ref_self']:.6f}（不可约下界）")

    results = {}

    def record(key, label, nfe, pc_gen, ckpt):
        m = gen_metrics(pc_gt, pc_gen, args)
        m = {"label": label, "nfe": nfe, "checkpoint": ckpt,
             "n_samples": int(pc_gen.shape[0]), **m}
        results[key] = m
        print(f"  {label:<26} CD={m['cd_mean']:.4f}  Cov@0.1={m['coverage_0.1']:.3f}"
              f"  Cov@0.2={m['coverage_0.2']:.3f}  1-NNA={m['nna_accuracy']:.3f}"
              f"  MMD-CD={m['mmd_cd']:.4f}")
        return m

    def ck(*names):
        """checkpoint 路径统一用 POSIX 分隔符落盘（跨平台可读）。"""
        return " + ".join(os.path.join(models["src"], n).replace(os.sep, "/")
                          for n in names)

    print("\n分布型指标（样本集规模 = 参考集规模，同测试批 / 同 CFG）:")
    for nfe in args.nfe_list:
        record(f"fm_nfe{nfe}", f"FMShape (Euler NFE={nfe})", nfe,
               sample_fm(models, cond, args, device, nfe),
               ck("condenc_fm_best.pth", "vnet_fm_best.pth"))

    record("ddpm_nfe100", f"DDPM (ancestral NFE={args.T})", args.T,
           sample_ddpm(models, cond, args, device),
           ck("condenc_best.pth", "epsnet_best.pth"))

    try:
        student, student_path = load_student(args, device)
        record("student_1step", "Distilled student (1 step)", 1,
               sample_student(models, student, cond, args, device),
               student_path.replace(os.sep, "/"))
    except FileNotFoundError as e:
        print(f"  [跳过] {e}")

    record("vae_oracle", "VAE oracle (GT latent decode)", 0,
           sample_vae_oracle(models, pc_gt), ck("vae_best.pth"))

    # ---------------- JSON 落盘 ----------------
    out = {
        "results": results,
        "reference": ref_stats,
        "metric_definitions": {
            "cd": "train.chamfer_distance_loss（双向最小平方距离均值，归一化点云单位），全指标复用，不重造",
            "coverage_tau": "参考点云中至少有一个生成样本 CD < tau 的比例；tau 与 F-Score 阈值一致",
            "nna_accuracy": "生成点云在 生成∪参考 混合池中的最近邻（排除自身）落在生成侧的比例；"
                            "≈0.5 = 与参考不可区分，>0.5 = 记忆化/模式坍塌信号；"
                            "配 pairwise_cd_gen_internal（生成集内部 i≠j CD）解读——"
                            "内部 CD 远小于 gen→ref CD 时，触顶反映多样性坍塌而非复制参考",
            "mmd_cd": "以 CD 为核的 MMD 估计 = 0.5·(mean CD(gen→ref) + mean CD(ref→gen))；"
                      "条件生成下 gen_i/ref_i 同条件配对，另报 mmd_cd_offdiag（i≠j）剔除配对角",
            "mmd_cd_ref_self": "参考集 i≠j 有序对 CD 均值：同分布两次独立抽样的不可约下界",
            "note": "样本集规模 = 参考集规模（n=%d，每参考条件 1 个样本），"
                    "覆盖率/1-NNA 在此规模下方差较大，读数按量级而非精确值解读"
                    % ref_stats["n"],
        },
        "protocol": {
            "task": "C1 HRRP-conditioned point-cloud reconstruction, common test batch",
            "ckpt_dir": args.ckpt_dir,
            "student_dir": args.student_dir,
            "cond_feat": args.cond_feat,
            "phase_mode": args.phase_mode,
            "seed": args.seed,
            "eval_seed": args.eval_seed,
            "sample_seed": args.sample_seed,
            "x0_seed": args.x0_seed,
            "test_data": args.test_data,
            "batch_size": args.batch_size,
            "n_eval": args.n_eval,
            "num_points": args.num_points,
            "T": args.T,
            "cfg_scale": args.cfg_scale,
            "nfe_list": args.nfe_list,
            "cd_taus": args.cd_taus,
            "torch": torch.__version__,
        },
    }
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "gen_metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n结果已保存: {path}")

    # 汇总表
    print("\n" + "=" * 92)
    print(f"{'模型':<28}{'CD↓':>9}{'Cov@0.1↑':>11}{'Cov@0.2↑':>11}"
          f"{'1-NNA':>9}{'MMD-CD↓':>11}")
    print("-" * 92)
    for k, m in results.items():
        print(f"{m['label']:<28}{m['cd_mean']:9.4f}{m['coverage_0.1']:11.3f}"
              f"{m['coverage_0.2']:11.3f}{m['nna_accuracy']:9.3f}{m['mmd_cd']:11.4f}")
    print("-" * 92)
    print(f"{'参考集自 MMD-CD (下界)':<28}{'':9}{'':11}{'':11}{'':9}"
          f"{ref_stats['mmd_cd_ref_self']:11.4f}")
    print("=" * 92)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分布型生成指标：Coverage / 1-NNA / MMD-CD")
    parser.add_argument("--ckpt_dir", type=str, default="./sat_model_c1",
                        help="C1 HRRP 条件模型目录（含 sat/ 子目录）")
    parser.add_argument("--student_dir", type=str, default="./sat_model_distill")
    parser.add_argument("--save_dir", type=str, default="./sat_model_cmp")
    parser.add_argument("--cond_feat", choices=["narrowband", "hrrp", "both", "isar"],
                        default="hrrp")
    parser.add_argument("--phase_mode", choices=["random", "tracked"], default="random")
    parser.add_argument("--test_data", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_eval", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--nfe_list", nargs="+", type=int, default=[1, 10],
                        help="FMShape ODE 步数（NFE）列表")
    parser.add_argument("--cd_taus", nargs="+", type=float, default=[0.1, 0.2],
                        help="Coverage / F-Score 阈值（归一化点云单位）")
    parser.add_argument("--tau", type=int, default=8, help="数据集帧数（与训练一致）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=999,
                        help="测试批重播种（与 train_fm_distill / verify_fm_distill_diversity 一致）")
    parser.add_argument("--sample_seed", type=int, default=999,
                        help="采样前全局播种（DDPM / FM 的初始噪声）")
    parser.add_argument("--x0_seed", type=int, default=999,
                        help="学生 x0 生成器种子（复现蒸馏评估约定）")
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
