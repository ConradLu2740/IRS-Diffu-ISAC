"""
phase_optimizer_sat.py — 星-地 ISAC 动态 RIS 相位优化

在动态几何下优化 RIS 相位（最大化接收功率），支持三种策略对比：
  - random:      随机相位（基线）
  - ideal:       逐帧优化（理想跟踪，update_interval=1）
  - segmented:   分段跟踪（每 update_interval 帧更新一次，模拟 RIS 重构速率限制）

核心洞察：LEO 信道相干时间 ms 级，RIS 重构速率受限时相位会"过期"，
分段跟踪实验量化了"跟踪精度 vs 硬件约束"的权衡——这是论文的关键图之一。

---------------------------------------------------------------------------
目标函数与相位设计的推导（v1.7 审查修正，2026-09-08）

评估目标 `_power`（与 data_sat 的 receive_signal 一致）中，接收信号对 RIS
相位向量 v_i = exp(j·phase_i) 是**线性的**：

    y_a(v) = d_a + Σ_i v_i · m_{i,a}        （a = 接收天线，i = RIS 单元）

其中：
  - d = 直达+散射路径（BS→ROI→UE，不经 RIS）在 X 下的接收向量 [BS_ant]；
  - m_i = 单元 i 经**两条** RIS 路径（BS→ROI→IRS→UE 与 BS→IRS→ROI→UE）
    的合成系数向量 [BS_ant]；
  - 功率 P(v) = Σ_a |y_a(v)|²。

旧实现 optimize_frame_1path 只对齐其中一条路径（BS→IRS→ROI→UE），且用
天线等权求和代替 X 加权内积、不含直达项参考 → 在 _power 同一目标下只达到
可实现的 ~53%（默认场景，坐标上升参考：+89.0% vs +256.1%）。

本模块默认使用**全模型闭式对齐**（optimize_frame）：固定 v 之外，单看
单元 i 与直达项的相干叠加，Σ_a |d_a + v_i·m_{i,a}|² 对 v_i 的闭式最优为

    v_i* = conj(d^H m_i) / |d^H m_i|   →   phase_i = −∠(d^H m_i)

（忽略单元间交叉项；N=16 时达数值参考 ~77%，见 audit/ris_objective_audit.py）。
真正的上界由 optimize_frame_numeric（坐标上升，多起点）给出，用于报告的
oracle/可达上限；分段跟踪 K 扫描默认用闭式对齐（方法），同时可输出数值
参考行验证权衡结论的稳健性。

约定说明：本仓库将 H_total [BS_ant, UE_ant] 与 X [UE_ant, 1] 相乘并沿
BS 天线求和取功率，等价于“UE 发、BS 收”的上行功率；X 为导频权重。
任何对“下行通信功率”的表述都应先说明此约定（收/发互易性），勿混用。
"""

import math
import torch
import numpy as np


def _as_complex(t):
    return torch.complex(t, torch.zeros_like(t))


class PhaseOptimizerSat:
    def __init__(self, channels, n_iter=20, device="cpu"):
        self.channels = channels          # data_sat.SatScenarioChannels
        self.n_iter = n_iter
        self.device = device

    # ------------------------------------------------------------------
    @staticmethod
    def _linear_model(Ht, ROI_voxel, X):
        """把 _power 目标线性化为 y(v) = d + M·v。

        返回 (d [BS_ant], M [BS_ant, N])：
          d   = (H_BS_ROI·S·H_ROI_UE)·X（直达+散射，v 无关）
          M   = 第 i 列为单元 i 的两条 RIS 路径合成系数 m_i
        """
        S = ROI_voxel.reshape(-1).float()
        S_c = _as_complex(S)
        if "H_ROI_IRS" not in Ht:
            return None, None
        H_BS_ROI, H_ROI_UE = Ht["H_BS_ROI"], Ht["H_ROI_UE"]
        Bmat = S_c[:, None] * H_ROI_UE                          # [R, UE]
        d = H_BS_ROI.matmul(Bmat).matmul(X).flatten()           # [BS]
        C1 = (H_BS_ROI * S_c[None, :]).matmul(Ht["H_ROI_IRS"])  # [BS,N] path1 前段
        B1 = (Ht["H_IRS_ROI"] * S_c[None, :]).matmul(H_ROI_UE)  # [N,UE] path2 后段
        HuX = Ht["H_IRS_UE"].matmul(X).flatten()                # [N]   path1 UE 端标量
        B1X = B1.matmul(X).flatten()                            # [N]   path2 UE 端标量
        # 单元 i 系数：m_i = C1[:,i]·HuX[i]（path1）+ H_BS_IRS[:,i]·B1X[i]（path2）
        M = C1 * HuX[None, :] + Ht["H_BS_IRS"] * B1X[None, :]    # [BS,N]
        return d, M

    # ------------------------------------------------------------------
    @torch.no_grad()
    def optimize_frame(self, Ht, ROI_voxel, X, phase_init=None):
        """对单帧信道 Ht 优化 RIS 相位（默认：全模型闭式对齐）。

        闭式解 v_i* = conj(d^H m_i)/|d^H m_i|，d^H m_i = Σ_a conj(d_a)·m_{i,a}。
        忽略单元间交叉项（跨单元耦合由数值参考 optimize_frame_numeric 兜底）。
        """
        d, M = self._linear_model(Ht, ROI_voxel, X)
        if d is None:
            return torch.tensor([], dtype=torch.float32)
        c = torch.conj(d)[None, :].matmul(M).flatten()          # [N]
        return (-torch.angle(c)).to(torch.float32)

    @torch.no_grad()
    def optimize_frame_1path(self, Ht, ROI_voxel, X, phase_init=None):
        """旧实现（保留对照）：只对齐 BS→IRS→ROI→UE 单条路径。

        用天线等权求和，不含直达项参考与第二条 RIS 路径；在 _power 同一
        目标下约只达全模型闭式解的 60~70%（默认场景 +89.0% vs +173.1%）。
        """
        if "H_ROI_IRS" not in Ht:
            return torch.tensor([])
        S = ROI_voxel.reshape(-1).float()
        S_c = _as_complex(S)
        A = (Ht["H_IRS_ROI"] * S_c[None, :]).matmul(Ht["H_ROI_UE"])   # [N, UE]
        g = Ht["H_BS_IRS"].sum(dim=0) * A.sum(dim=1)                  # [N]
        return (-torch.angle(g)).to(torch.float32)

    @torch.no_grad()
    def optimize_frame_numeric(self, Ht, ROI_voxel, X, n_restart=6,
                               max_sweeps=25, seed=0):
        """数值参考上界：完整目标下坐标上升（每单元闭式更新，多起点）。

        返回 (phase [N], power float)。仅用于 oracle/可达上限报告与审计，
        不作为低开销方法使用。
        """
        d, M = self._linear_model(Ht, ROI_voxel, X)
        if d is None:
            return torch.tensor([]), 0.0
        N = M.shape[1]
        gen = torch.Generator().manual_seed(seed)
        starts = [torch.zeros(N)]
        starts += [torch.rand(N, generator=gen) * 2 * math.pi for _ in range(n_restart)]
        best_p, best_phase = -1.0, None
        for ph in starts:
            v = torch.exp(1j * ph).to(torch.complex64)
            for _sweep in range(max_sweeps):
                y = d + M.matmul(v)
                for i in range(N):
                    mi = M[:, i]
                    r = y - v[i] * mi
                    c = torch.sum(torch.conj(r) * mi)
                    v_new = torch.conj(c) / (torch.abs(c) + 1e-12)
                    if torch.abs(c) < 1e-12:
                        v_new = torch.tensor(1.0 + 0j)
                    y = r + v_new * mi
                    v[i] = v_new
            p = float(torch.sum(torch.abs(y) ** 2).item())
            if p > best_p:
                best_p, best_phase = p, torch.angle(v)
        return best_phase.to(torch.float32), best_p

    # ------------------------------------------------------------------
    def _power(self, Ht, ROI_voxel, X, phase):
        """给定相位计算接收功率 |Y|²（标量）。"""
        v = torch.exp(1j * phase).to(torch.complex64)
        S = ROI_voxel.reshape(-1).float()
        S_c = _as_complex(S)

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
                     n_iter=10, intervals=(1, 2, 4, 8),
                     include_numeric=False):
    """对比 随机 / 理想跟踪 / 分段跟踪 的平均接收功率。

    默认策略为全模型闭式对齐（optimize_frame）。include_numeric=True 时
    额外输出 numeric_K=1 / numeric_K=8（坐标上升上界参考）与 closed-form
    达成数值上界的百分比，用于报告可达上限。
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

    # 理想/分段跟踪（闭式对齐）
    for k in intervals:
        phases, powers = opt.optimize_sequence(ROI_voxel, X, update_interval=k)
        powers = [p for p in powers if p > 0]  # 只统计有 IRS 的帧
        results[f"track_K={k}"] = {"power": float(np.mean(powers)), "phases": phases}

    # 数值参考上界（坐标上升）
    if include_numeric:
        def _numeric_seq(update_interval):
            frames = channels.frames
            current, powers, t_last = None, [], None
            for t in range(len(frames)):
                Ht = channels.channels_per_frame[t]
                if "H_ROI_IRS" not in Ht:
                    continue
                if t % update_interval == 0:
                    current, _ = opt.optimize_frame_numeric(Ht, ROI_voxel, X, seed=t)
                powers.append(opt._power(Ht, ROI_voxel, X, current))
            return powers
        p_n1 = _numeric_seq(1)
        p_n8 = _numeric_seq(8)
        results["numeric_K=1"] = {"power": float(np.mean(p_n1)), "phases": []}
        results["numeric_K=8"] = {"power": float(np.mean(p_n8)), "phases": []}

    return results


if __name__ == "__main__":
    pass
