"""audit_ris_doppler.py — Task A 可行性：RIS 多普勒解耦门（spec 2026-09-08）

量化三件事，用于判断"RIS 多普勒域感知-通信解耦"是否可行（继续 or 止损）：
  1. 感知回波经 RIS（BS→ROI→RIS→UE）功率 vs 通信经 RIS（BS→RIS→UE）功率；
     两者都被 RIS 反射（慢时斜坡/共相）。比值决定"注入多普勒分离是否有意义"。
  2. 直达感知（BS→ROI→UE，无 RIS，现有感知基线）功率作为参照。
  3. 慢时多普勒可行性：帧率/无模糊/目标径向多普勒 vs 注入 Δf_d 的排序关系。

物理直觉（需数据证实）：
  - sat 模式（RIS 挂卫星旁）：回波多两条 ~695 km 腿 → 可能远弱于通信；
  - ground 模式（RIS 在 UE 旁）：回波只多一条 ~47.7 km 腿 → 可行性更高。
"""
import math
import os
import sys

import numpy as np
import torch

import setup_sat as ss
from data_sat import (SatScenarioChannels, generate_ground_target_sample,
                      _SIGNAL1)
from phase_optimizer_sat import PhaseOptimizerSat

torch.manual_seed(42)
np.random.seed(42)


def main():
    scenario = ss.SatISACScenario()
    frames = scenario.build_frames()
    mid = frames[len(frames) // 2]
    roi, cid, ang = generate_ground_target_sample()
    S = torch.tensor(roi.reshape(-1), dtype=torch.float32)

    print("=== 慢时多普勒可行性 ===")
    dt = scenario.frame_interval_s
    n_frames = len(frames)
    print(f"帧间隔 dt={dt} s；帧数={n_frames}"
          f"；slow-time Nyquist=±{0.5/dt:g} Hz；分辨率≈{1/(n_frames*dt):g} Hz")
    # 目标径向多普勒（地面目标相对卫星，速度量级假设 1-30 m/s，λ=0.01m）
    lam = scenario.wavelength_m
    for v in (1.0, 10.0, 30.0):
        print(f"  {v:4.1f} m/s 目标径向 → 多普勒 {v/lam:.0f} Hz"
              f"（远超 Nyquist {0.5/dt:g} Hz，会严重混叠）")

    print("\n=== 各路径功率（共相对齐相位，v1.7 闭式） ===")
    for mode in ("sat", "ground"):
        ch = SatScenarioChannels(frames, irs_mode=mode, device="cpu")
        X = ch.tensor_a * torch.tensor(_SIGNAL1[:4], dtype=torch.complex64).view(4, 1)
        S_c = torch.complex(S, torch.zeros_like(S))
        Ht = ch.channels_per_frame[len(channels_per_frame := frames) // 2]
        # 用与 _power 一致的路径
        opt = PhaseOptimizerSat(ch)
        v = opt.optimize_frame(Ht, S, X)

        # 通信经 RIS：BS→RIS→UE
        H_BS_ROI, H_ROI_UE = Ht["H_BS_ROI"], Ht["H_ROI_UE"]
        H_BS_IRS, H_ROI_IRS = Ht["H_BS_IRS"], Ht["H_ROI_IRS"]
        H_IRS_ROI, H_IRS_UE = Ht["H_IRS_ROI"], Ht["H_IRS_UE"]
        vv = torch.exp(1j * v).to(torch.complex64)

        # comm = (H_BS_IRS * vv) @ H_IRS_UE @ X
        comm = (H_BS_IRS * vv[None, :]).matmul(H_IRS_UE)
        P_comm = float(torch.sum(torch.abs(comm.matmul(X)) ** 2).item())

        # echo via RIS = (H_BS_ROI·S·H_ROI_IRS * vv) @ H_IRS_UE @ X
        C1 = (H_BS_ROI * S_c[None, :]).matmul(H_ROI_IRS)
        echo = ((C1 * vv[None, :]).matmul(H_IRS_UE))
        P_echo = float(torch.sum(torch.abs(echo.matmul(X)) ** 2).item())

        # direct sensing = H_BS_ROI·S·H_ROI_UE @ X (无 RIS)
        P_sense_dir = float(torch.sum(
            torch.abs(H_BS_ROI.matmul(S_c[:, None] * H_ROI_UE).matmul(X)) ** 2).item())

        print(f"[{mode:>6}] comm(RIS)={P_comm:.4e}  echo(RIS)={P_echo:.4e}  "
              f"sense_direct={P_sense_dir:.4e}")
        print(f"        echo/comm = {P_echo/max(P_comm,1e-18):.3e}"
              f"   sense_direct/comm = {P_sense_dir/max(P_comm,1e-18):.3e}")


if __name__ == "__main__":
    main()
