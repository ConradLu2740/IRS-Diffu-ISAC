"""Flow Matching（OT-CFM）冒烟测试。

运行：python tests/test_smoke_fm.py  （或 pytest tests/）
原则：小样本、秒级、CPU 可跑；验证 CFM 数学、训练收敛与 ODE 采样接口。
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

_ISAC = Path(__file__).resolve().parents[1] / "source_code" / "isac_sat"
_LEGACY = Path(__file__).resolve().parents[1] / "source_code" / "legacy"
sys.path.insert(0, str(_ISAC))
sys.path.insert(0, str(_LEGACY))

from fm_utils import CFMScheduler, train_1D_FM, sample_conditional_FM  # noqa: E402
from models import PointVAE, AdvancedCondEncoder, LatentDiT1D_CrossAttn  # noqa: E402
from train import estimate_latent_stats, chamfer_distance_loss  # noqa: E402


def _make_toy(seed=42, n=16, num_points=64, tau=8, cond_dim=16):
    g = torch.Generator().manual_seed(seed)
    pcs = torch.randn(n, num_points, 3, generator=g) * 0.3
    conds = torch.randn(n, tau, cond_dim, generator=g)
    return pcs, conds


def test_cfm_interpolation_endpoints():
    """OT-CFM 插值：t=0 还原噪声端，t=1 还原数据端，目标速度恒为 x1-x0。"""
    sched = CFMScheduler()
    x0 = torch.randn(4, 8)
    x1 = torch.randn(4, 8)
    xt0, v0 = sched.interpolate(x0, x1, torch.zeros(4))
    xt1, v1 = sched.interpolate(x0, x1, torch.ones(4))
    assert torch.allclose(xt0, x0)
    assert torch.allclose(xt1, x1)
    assert torch.allclose(v0, x1 - x0) and torch.allclose(v1, x1 - x0)


def test_fm_train_and_sample(tmp_path=None):
    """FM 全链路：训练损失下降 + ODE 采样输出形状/数值合法。"""
    torch.manual_seed(42)
    device = "cpu"
    pcs, conds = _make_toy()
    loader = DataLoader(TensorDataset(pcs, conds), batch_size=8, shuffle=False)

    vae = PointVAE(num_points=64, z_dim=256).to(device)
    condenc = AdvancedCondEncoder(seq_len=8, input_size=16, hidden_size=128,
                                  out_emb=256).to(device)
    vnet = LatentDiT1D_CrossAttn(z_dim=256, cond_emb=256, hidden_size=256,
                                 depth=1, num_heads=8).to(device)
    z_mean, z_std = estimate_latent_stats(vae, loader, device=device)

    save_dir = str(tmp_path) if tmp_path is not None else "/tmp/fm_smoke"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    hist = train_1D_FM(vae, condenc, vnet, loader, loader, z_mean, z_std,
                       device=device, epochs=15, lr_v=1e-3, save_dir=save_dir)
    assert hist["train_total"][-1] < 0.8 * hist["train_total"][0], \
        f"FM loss 未下降: {hist['train_total']}"

    cond = conds[:4]
    pc_gt = pcs[:4]
    for nfe in (1, 10):
        pc_hat = sample_conditional_FM(vae, condenc, vnet, cond, z_mean, z_std,
                                       device=device, cfg_scale=2.0, nfe=nfe)
        assert pc_hat.shape == pc_gt.shape
        assert torch.isfinite(pc_hat).all()
        cd = chamfer_distance_loss(pc_gt, pc_hat)
        assert torch.isfinite(cd), f"NFE={nfe} CD 非法: {cd}"

    pc_a = sample_conditional_FM(vae, condenc, vnet, cond, z_mean, z_std,
                                 device=device, cfg_scale=2.0, nfe=1)
    pc_b = sample_conditional_FM(vae, condenc, vnet, cond, z_mean, z_std,
                                 device=device, cfg_scale=2.0, nfe=10)
    assert not torch.allclose(pc_a, pc_b), "不同 NFE 采样结果完全相同，ODE 积分疑似失效"


TESTS = [test_cfm_interpolation_endpoints, test_fm_train_and_sample]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    if failed:
        sys.exit(f"{failed} test(s) failed")
    print(f"ALL PASS ({len(TESTS)} FM smoke tests)")
