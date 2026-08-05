"""
phase_optimizer_sat.py — 星-地 ISAC 动态 RIS 相位优化

在动态几何下优化 RIS 相位（最大化接收功率），支持三种策略对比：
  - random:      随机相位（当前 data_sat.py 的行为）
  - ideal:       逐帧优化（理想跟踪，update_interval=1）
  - segmented:   分段跟踪（每 update_interval 帧更新一次，模拟 RIS 重构速率限制）

核心洞察：LEO 信道相干时间 ms 级，RIS 重构速率受限时相位会"过期"，
分段跟踪实验量化了"跟踪精度 vs 硬件约束"的权衡——这是论文的关键图之一。
"""

import math
import torch
import numpy as np


class PhaseOptimizerSat:
    def __init__(self, channels, n_iter=20, device="cpu"):
        self.channels = channels          # data_sat.SatScenarioChannels
        self.n_iter = n_iter
        self.device = device

    # ------------------------------------------------------------------
    @torch.no_grad()
    def optimize_frame(self, Ht, ROI_voxel, X, phase_init=None):
        """对单帧信道 Ht 优化 RIS 相位。

        使用解析相位对齐：让每个 IRS 元素的相干合成信道相位对齐
        （最大化接收功率的闭式解，优于坐标上升）。
        返回 [N] 相位（弧度）。
        """
        if "H_ROI_IRS" not in Ht:
            return torch.tensor([])
        S = ROI_voxel.reshape(-1).float()
        S_c = torch.complex(S, torch.zeros_like(S)).to(self.device)
        # ROI 散射后的中间信道: A[i,b] = sum_j H_IRS_ROI[i,j] * S_j * H_ROI_UE[j,b]
        A = (Ht["H_IRS_ROI"] * S_c[None, :]).matmul(Ht["H_ROI_UE"])   # [N, UE]
        # 每个 IRS 元素的有效相干增益（对全部 BS/UE 天线求和）:
        #   g_i = (sum_a H_BS_IRS[a,i]) * (sum_b A[i,b])
        g = Ht["H_BS_IRS"].sum(dim=0) * A.sum(dim=1)                    # [N]
        return -torch.angle(g)  # 相位对齐

    # ------------------------------------------------------------------
    def _power(self, Ht, ROI_voxel, X, phase):
        """给定相位计算接收功率 |Y|²（标量）。"""
        v = torch.exp(1j * phase).to(torch.complex64)
        S = ROI_voxel.reshape(-1).float()
        S_c = torch.complex(S, torch.zeros_like(S)).to(self.device)

        H_BS_ROI = Ht["H_BS_ROI"]
        H_ROI_UE = Ht["H_ROI_UE"]
        Bmat = S_c[:, None] * H_ROI_UE
        H_total = H_BS_ROI.matmul(Bmat)                      # 直达

        C = H_BS_ROI * S_c[None, :]                          # BS→ROI→IRS→UE
        C1 = C.matmul(Ht["H_ROI_IRS"])
        H_total = H_total + (C1 * v[None, :]).matmul(Ht["H_IRS_UE"])

        IR1 = Ht["H_IRS_ROI"] * S_c[None, :]                 # BS→IRS→ROI→UE
        B1 = IR1.matmul(H_ROI_UE)
        H_total = H_total + (Ht["H_BS_IRS"] * v[None, :]).matmul(B1)

        Y = H_total.matmul(X)
        return torch.sum(torch.abs(Y) ** 2).item()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def optimize_sequence(self, ROI_voxel, X, update_interval=1):
        """对整个 Tau 帧序列做分段跟踪。

        update_interval=1 → 逐帧理想跟踪；=K → 每 K 帧更新一次相位。
        返回 (phases [Tau, N], powers [Tau])
        """
        frames = self.channels.frames
        phases, powers = [], []
        current = None
        for t in range(len(frames)):
            Ht = self.channels.channels_per_frame[t]
            if "H_ROI_IRS" not in Ht:
                phases.append(torch.tensor([]))
                powers.append(0.0)
                continue
            if t % update_interval == 0:
                current = self.optimize_frame(Ht, ROI_voxel, X)
            phases.append(current.clone())
            powers.append(self._power(Ht, ROI_voxel, X, current))
        return phases, powers

    # ------------------------------------------------------------------
    def random_phase_sequence(self, n_frames, n_irs):
        return [torch.rand(n_irs, device=self.device) * 2 * math.pi
                for _ in range(n_frames)]


def compare_tracking(channels, ROI_voxel, X, device="cpu",
                     n_iter=10, intervals=(1, 2, 4, 8)):
    """对比 随机 / 理想跟踪 / 分段跟踪 的平均接收功率。

    返回 dict: {策略名: {"power": float, "phases": list}}
    """
    opt = PhaseOptimizerSat(channels, n_iter=n_iter, device=device)
    results = {}

    # 随机相位基线
    n_irs = channels.channels_per_frame[0]["H_ROI_IRS"].shape[1] \
            if "H_ROI_IRS" in channels.channels_per_frame[0] else 0
    rand_phases = opt.random_phase_sequence(len(channels.frames), n_irs)
    rand_powers = []
    for t, Ht in enumerate(channels.channels_per_frame):
        if "H_ROI_IRS" not in Ht:
            continue
        rand_powers.append(opt._power(Ht, ROI_voxel, X, rand_phases[t]))
    results["random"] = {"power": float(np.mean(rand_powers)), "phases": rand_phases}

    # 理想/分段跟踪
    for k in intervals:
        phases, powers = opt.optimize_sequence(ROI_voxel, X, update_interval=k)
        powers = [p for p in powers if p > 0]  # 只统计有 IRS 的帧
        results[f"track_K={k}"] = {"power": float(np.mean(powers)), "phases": phases}

    return results
