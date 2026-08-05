"""
demo_sdr.py — SDR 数据管线演示（无硬件，文件回放）

流程：
  1. 生成模拟 IQ（ISAC 仿真 → 时域回波 IQ）→ 存文件（sdr_io 格式）
  2. 加载 IQ → sdr_ingest 恢复距离像（FFT 管线）
  3. 保真验证：仿真距离像 vs SDR 管线距离像
  4. 感知：SDR 距离像 → 目标分类 + 定位

有硬件后：sdr_io.capture_live() 采集真实 IQ → 同一 sdr_ingest 管线。

用法：
  python demo_sdr.py [--checkpoint ./isac_demo/sensing_best.pth]
"""

import os
import argparse
import numpy as np
import torch

import setup_sat as ss
from data_sat import (SatScenarioChannels, GROUND_TARGET_TEMPLATES,
                      generate_ground_target_sample)
from sdr_io import save_iq, load_iq, simulate_isac_iq_wideband
from sdr_ingest import iq_to_range_profile, verify_pipeline
from train_sensing import SensingMLP

CLASS_NAMES = [n for n, _ in GROUND_TARGET_TEMPLATES]
OUT_DIR = "./isac_demo"
os.makedirs(OUT_DIR, exist_ok=True)


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device=device,
                                   bs_ant=args.bs_ant, ue_ant=args.ue_ant)
    mid = frames[len(frames) // 2]

    # ---- 1. 模拟 IQ 生成 + 保存 ----
    roi, cid_true, ang = generate_ground_target_sample()
    pos_true = np.argwhere(roi > 0.5).astype(np.float32).mean(axis=0)
    pos_true = (pos_true / 16.0 * 2.0 - 1.0)[:2]

    iq = simulate_isac_iq_wideband(roi, mid["target_pos"], mid["ground_pos"],
                                   channels.wavelength_m, snr_db=None, seed=args.seed)
    iq_path = os.path.join(OUT_DIR, "sample_isac.iq.npz")
    save_iq(iq, iq_path, fs_hz=1e9, fc_hz=scenario.fc_hz,
            utc="2026-08-05T12:00:00Z", meta={"source": "simulated"})
    print(f"[sdr] 模拟 IQ 已保存: {iq_path} ({len(iq)} 采样)")

    # ---- 2. 保真验证（仿真 vs SDR 管线） ----
    corr, sim_rp, sdr_rp = verify_pipeline(
        roi, mid["target_pos"], mid["ground_pos"], channels.wavelength_m,
        snr_db=args.snr_db, seed=args.seed)
    print(f"[sdr] 管线保真: 仿真距离像 vs SDR恢复 相关 = {corr:.4f} "
          f"{'(PASS)' if corr > 0.9 else '(CHECK)'}")

    # ---- 3. 加载 IQ → SDR 距离像 → 感知 ----
    iq2, meta = load_iq(iq_path)
    rp = iq_to_range_profile(iq2, snr_db=args.snr_db, seed=args.seed + 1, align=False)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SensingMLP(in_dim=ckpt["feat_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        logits, pred_pos = model(torch.from_numpy(rp).float().unsqueeze(0))
    cid_pred = logits.argmax(1).item()
    pos_pred = pred_pos[0].numpy()[:2]
    cls_ok = cid_pred == cid_true
    pos_err = float(np.linalg.norm(pos_pred - pos_true))

    print("\n" + "=" * 58)
    print("SDR 管线感知结果:")
    print(f"  真实目标 : {CLASS_NAMES[cid_true]} @ ({pos_true[0]:+.2f},{pos_true[1]:+.2f})")
    print(f"  SDR 感知 : {CLASS_NAMES[cid_pred]} @ ({pos_pred[0]:+.2f},{pos_pred[1]:+.2f})")
    print(f"  分类 {'✓' if cls_ok else '✗'} · 定位误差 {pos_err:.3f}")
    print("=" * 58)
    print(f"[sdr] 元数据: fs={meta['fs_hz']/1e6:.0f}MHz fc={meta['fc_hz']/1e9:.1f}GHz utc={meta['utc']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDR 数据管线演示")
    parser.add_argument("--checkpoint", type=str, default="./isac_demo/sensing_best.pth")
    parser.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    parser.add_argument("--bs_ant", type=int, default=4)
    parser.add_argument("--ue_ant", type=int, default=4)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
