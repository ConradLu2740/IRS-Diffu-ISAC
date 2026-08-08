"""
demo_live.py — 星-地 ISAC 感知-通信闭环实时演示（数据生成器 + HTML 播放器）

跑多次卫星过境仿真（不同 seed/目标），生成单文件 HTML 播放器：
  - 顶部下拉：切换场景（不同目标/感知结果）
  - 动画 + 时间轴 + 播放控制
  - 每帧含真实过境 UTC 时间

用法：
  python demo_live.py [--checkpoint ./isac_demo/sensing_best.pth] [--n_scenes 3]
  输出：./isac_demo/demo_live.html（浏览器直接打开）
"""

import os
import json
import argparse
from datetime import datetime, timedelta
import numpy as np
import random
import torch

import setup_sat as ss
from data_sat import (SatScenarioChannels, GROUND_TARGET_TEMPLATES,
                      compute_range_profile, _SIGNAL1, generate_ground_target_sample)
from phase_optimizer_sat import PhaseOptimizerSat
from train_sensing import SensingMLP

CLASS_NAMES = [n for n, _ in GROUND_TARGET_TEMPLATES]
OUT_DIR = "./isac_demo"
os.makedirs(OUT_DIR, exist_ok=True)


def run_scene(args, seed):
    """跑一个场景，返回单场景 payload。"""
    torch.manual_seed(seed)
    random.seed(seed); np.random.seed(seed)
    device = args.device

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SensingMLP(in_dim=ckpt["feat_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scenario = ss.SatISACScenario(tau=args.tau)
    frames = scenario.build_frames()
    channels = SatScenarioChannels(frames, irs_mode=args.irs_mode, device=device,
                                   bs_ant=args.bs_ant, ue_ant=args.ue_ant)
    mid = frames[len(frames) // 2]
    opt = PhaseOptimizerSat(channels, device=device)
    X = channels.tensor_a * torch.tensor(_SIGNAL1[:channels.bs_ant],
                                         dtype=torch.complex64).view(channels.bs_ant, 1)

    roi, cid_true, ang_true = generate_ground_target_sample()
    pos_true = np.argwhere(roi > 0.5).astype(np.float32).mean(axis=0)
    pos_true = (pos_true / 16.0 * 2.0 - 1.0)[:2].tolist()

    rp = compute_range_profile(roi, mid["target_pos"], mid["ground_pos"],
                               channels.wavelength_m, snr_db=args.snr_db, seed=0,
                               align=False)
    with torch.no_grad():
        logits, pred_pos = model(torch.from_numpy(rp).float().unsqueeze(0))
    cid_pred = logits.argmax(1).item()
    pos_pred = pred_pos[0].numpy()[:2].tolist()
    cls_ok = cid_pred == cid_true
    pos_err = float(np.linalg.norm(np.array(pos_pred) - np.array(pos_true)))

    res = 16
    roi_est = np.zeros((res, res, res), dtype=np.float32)
    cx = int((pos_pred[0] + 1) / 2 * res); cy = int((pos_pred[1] + 1) / 2 * res)
    roi_est[max(cx - 2, 0):min(cx + 3, res), max(cy - 2, 0):min(cy + 3, res), 7:9] = 1.0
    roi_true_t = torch.tensor(roi.astype(np.float32)).reshape(-1)
    roi_est_t = torch.tensor(roi_est.astype(np.float32)).reshape(-1)

    n_frames = len(frames)
    elevs = [f["elevation_deg"] for f in frames]
    powers = {"random": [], "sensed": [], "oracle": []}
    for Ht in channels.channels_per_frame:
        n_irs = Ht["H_ROI_IRS"].shape[1]
        powers["random"].append(opt._power(Ht, roi_true_t, X, torch.rand(n_irs) * 2 * np.pi))
        powers["sensed"].append(opt._power(Ht, roi_true_t, X, opt.optimize_frame(Ht, roi_est_t, X)))
        powers["oracle"].append(opt._power(Ht, roi_true_t, X, opt.optimize_frame(Ht, roi_true_t, X)))

    all_p = [p for v in powers.values() for p in v]
    p_min, p_max = min(all_p), max(all_p)
    p_norm = {k: [(p - p_min) / (p_max - p_min + 1e-9) for p in v]
              for k, v in powers.items()}
    gain_sensed = 100 * (np.mean(powers["sensed"]) / np.mean(powers["random"]) - 1)
    gain_oracle = 100 * (np.mean(powers["oracle"]) / np.mean(powers["random"]) - 1)

    frame_data = []
    t0 = datetime(*scenario.start_utc)
    for i in range(n_frames):
        utc = (t0 + timedelta(seconds=frames[i]["t_abs_sec"])).strftime("%H:%M:%S")
        frame_data.append({
            "t": i, "utc": utc, "elev": round(elevs[i], 1),
            "progress": round(i / max(n_frames - 1, 1), 3),
            "pr": round(p_norm["random"][i], 3),
            "ps": round(p_norm["sensed"][i], 3),
            "po": round(p_norm["oracle"][i], 3),
        })

    return {
        "meta": {
            "sat": scenario.sat_name, "freq": f"{scenario.fc_hz/1e9:.1f} GHz",
            "frames": n_frames, "target_lat": scenario.target_lat,
            "target_lon": scenario.target_lon, "irs": args.irs_mode,
        },
        "target": {
            "cls_true": CLASS_NAMES[cid_true], "cls_pred": CLASS_NAMES[cid_pred],
            "cls_ok": cls_ok,
            "pos_true": [round(v, 2) for v in pos_true],
            "pos_pred": [round(v, 2) for v in pos_pred],
            "pos_err": round(pos_err, 3),
        },
        "frames": frame_data,
        "summary": {"gain_sensed": round(gain_sensed, 1),
                    "gain_oracle": round(gain_oracle, 1)},
    }


def main(args):
    scenes = [run_scene(args, args.seed + i) for i in range(args.n_scenes)]
    payload = {"scenes": scenes}
    html = render_html(payload)
    out_path = os.path.join(OUT_DIR, "demo_live.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    for i, s in enumerate(scenes):
        t = s["target"]
        print(f"[scene {i}] {t['cls_pred']} (正确:{t['cls_ok']}) 位置误差 {t['pos_err']} "
              f"增益 {s['summary']['gain_sensed']:+.1f}%")
    print(f"[demo_live] HTML: {out_path}")


def render_html(payload):
    data_js = json.dumps(payload, ensure_ascii=False)
    return HTML_TEMPLATE.replace("__DATA__", data_js)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>星-地 ISAC 感知-通信闭环 · 实时演示</title>
<style>
  body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
         background: #0f1420; color: #e8ecf4; margin: 0; padding: 16px; }
  h1 { font-size: 20px; margin: 4px 0 12px; color: #7fd1ff; }
  .meta { font-size: 13px; color: #9aa7bd; margin-bottom: 12px; }
  .topbar { display: flex; align-items: center; gap: 14px; margin-bottom: 12px; flex-wrap: wrap; }
  .topbar select, .topbar label { font-size: 14px; }
  .topbar select { background: #1a2233; color: #e8ecf4; border: 1px solid #2a3550;
                   border-radius: 6px; padding: 6px 10px; }
  .grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 12px; }
  .card { background: #1a2233; border: 1px solid #2a3550; border-radius: 10px; padding: 12px; }
  .card h2 { font-size: 14px; margin: 0 0 8px; color: #b8c6de; }
  canvas { width: 100%; height: auto; background: #0d1320; border-radius: 8px; }
  .panel { display: flex; flex-direction: column; gap: 10px; }
  .stat { display: flex; justify-content: space-between; font-size: 14px; padding: 6px 8px;
          background: #141c2c; border-radius: 6px; }
  .stat .v { color: #7fd1ff; font-weight: bold; }
  .controls { display: flex; align-items: center; gap: 12px; margin-top: 12px; }
  button { background: #2a67c4; border: none; color: #fff; padding: 8px 16px; border-radius: 6px;
           cursor: pointer; font-size: 14px; }
  button:hover { background: #3a77d4; }
  input[type=range] { flex: 1; }
  .legend { display: flex; gap: 14px; font-size: 12px; color: #9aa7bd; flex-wrap: wrap; }
  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }
  @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
  <h1>🛰️ 星-地 ISAC 感知-通信闭环 · 实时演示</h1>
  <div class="topbar">
    <label>场景：<select id="sceneSel"></select></label>
    <span class="meta" id="meta"></span>
  </div>
  <div class="grid">
    <div class="card">
      <h2>卫星过境场景</h2>
      <canvas id="scene" width="640" height="400"></canvas>
      <div class="legend">
        <span><span class="dot" style="background:#4fc3f7"></span>卫星</span>
        <span><span class="dot" style="background:#7fd151"></span>目标</span>
        <span><span class="dot" style="background:#ffb74d"></span>地面站</span>
        <span><span class="dot" style="background:#ce93d8"></span>IRS 波束</span>
      </div>
    </div>
    <div class="card">
      <h2>实时感知与状态</h2>
      <div class="panel" id="info"></div>
    </div>
  </div>
  <div class="card" style="margin-top:12px">
    <h2>通信功率（三种 IRS 策略）</h2>
    <canvas id="power" width="980" height="180"></canvas>
  </div>
  <div class="controls">
    <button id="playBtn">⏸ 暂停</button>
    <button id="speedBtn">1.0×</button>
    <input type="range" id="timeline" min="0" value="0">
    <span id="timeLabel" style="font-size:13px">帧 0</span>
  </div>
<script>
const SCENES = __DATA__.scenes;
let DATA = SCENES[0], frames = DATA.frames, meta = DATA.meta;
let idx = 0, playing = true, speed = 1.0, timer = null;

// 场景下拉
const sel = document.getElementById('sceneSel');
SCENES.forEach((s, i) => {
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = `场景 ${i+1} · ${s.target.cls_true} (${s.target.cls_ok?'✓':'✗'}) · +${s.summary.gain_sensed}%`;
  sel.appendChild(opt);
});
sel.onchange = () => { DATA = SCENES[parseInt(sel.value)]; frames = DATA.frames; meta = DATA.meta; idx = 0; step(); };
function updateMeta() {
  document.getElementById('meta').textContent =
    `卫星: ${meta.sat} · 载频: ${meta.freq} · 目标区域: (${meta.target_lat}°N, ${meta.target_lon}°E) · IRS: ${meta.irs}`;
}

const scene = document.getElementById('scene'), sctx = scene.getContext('2d');
const power = document.getElementById('power'), pctx = power.getContext('2d');
const info = document.getElementById('info');

function drawScene(i) {
  const f = frames[i], W = scene.width, H = scene.height;
  sctx.clearRect(0, 0, W, H);
  const skyY = H * 0.42, groundY = H * 0.78;
  const grad = sctx.createLinearGradient(0, 0, 0, groundY);
  grad.addColorStop(0, '#0a1226'); grad.addColorStop(1, '#16233d');
  sctx.fillStyle = grad; sctx.fillRect(0, 0, W, groundY);
  sctx.fillStyle = '#1d2c1d'; sctx.fillRect(0, groundY, W, H - groundY);
  sctx.strokeStyle = '#3a5a3a'; sctx.lineWidth = 2;
  sctx.beginPath(); sctx.moveTo(0, groundY); sctx.lineTo(W, groundY); sctx.stroke();
  // 地面站
  sctx.fillStyle = '#ffb74d';
  sctx.beginPath(); sctx.arc(W*0.55, groundY-4, 7, 0, 2*Math.PI); sctx.fill();
  sctx.fillStyle = '#9aa7bd'; sctx.font = '11px sans-serif';
  sctx.fillText('地面站', W*0.55-20, groundY+16);
  // 目标（真实）
  const tpos = DATA.target.pos_true;
  const tx = W*0.45 + tpos[0]*30, ty = groundY - 4;
  sctx.fillStyle = '#7fd151';
  sctx.beginPath(); sctx.arc(tx, ty, 8, 0, 2*Math.PI); sctx.fill();
  sctx.fillText(DATA.target.cls_true, tx-14, ty-14);
  // 感知位置
  const pp = DATA.target.pos_pred;
  const pxx = W*0.45 + pp[0]*30, pyy = groundY - 4;
  sctx.strokeStyle = '#ff5252'; sctx.lineWidth = 2;
  sctx.beginPath(); sctx.moveTo(pxx-6, pyy-6); sctx.lineTo(pxx+6, pyy+6);
  sctx.moveTo(pxx+6, pyy-6); sctx.lineTo(pxx-6, pyy+6); sctx.stroke();
  // 卫星
  const prog = f.progress, elev = f.elev;
  const satX = W * (0.12 + 0.76 * prog);
  const satY = skyY + (1 - elev / 60) * (groundY - skyY - 30);
  sctx.fillStyle = '#4fc3f7';
  sctx.beginPath(); sctx.arc(satX, satY, 9, 0, 2*Math.PI); sctx.fill();
  sctx.fillText(meta.sat, satX-18, satY-16);
  // 轨迹
  sctx.strokeStyle = 'rgba(79,195,247,0.4)'; sctx.lineWidth = 2;
  sctx.setLineDash([5, 5]);
  sctx.beginPath();
  for (let j = 0; j <= idx; j++) {
    const p = frames[j].progress, e = frames[j].elev;
    const x = W*(0.12+0.76*p), y = skyY + (1 - e/60)*(groundY-skyY-30);
    j === 0 ? sctx.moveTo(x, y) : sctx.lineTo(x, y);
  }
  sctx.stroke(); sctx.setLineDash([]);
  // 感知链路
  sctx.strokeStyle = 'rgba(255,82,82,0.5)'; sctx.lineWidth = 1.5;
  sctx.beginPath(); sctx.moveTo(satX, satY); sctx.lineTo(tx, ty); sctx.stroke();
  // IRS 波束
  if (f.ps > 0.2) {
    sctx.strokeStyle = 'rgba(206,147,216,0.6)'; sctx.lineWidth = 3;
    sctx.beginPath();
    sctx.moveTo(W*0.55, groundY-4);
    sctx.lineTo(tx-10, groundY-60); sctx.lineTo(tx+10, groundY-60);
    sctx.closePath(); sctx.stroke();
    sctx.fillStyle = 'rgba(206,147,216,0.15)'; sctx.fill();
  }
  sctx.fillStyle = '#7fd1ff'; sctx.font = '12px sans-serif';
  sctx.fillText(`仰角 ${elev.toFixed(1)}° · UTC ${f.utc} · 帧 ${i}`, 10, 22);
}

function drawPower(i) {
  const W = power.width, H = power.height;
  pctx.clearRect(0, 0, W, H);
  const colors = { pr: '#9aa7bd', ps: '#ff5252', po: '#4fc3f7' };
  const labels = { pr: 'random', ps: 'sensed', po: 'oracle' };
  const names = ['pr', 'ps', 'po'];
  const pad = 30;
  names.forEach(n => {
    pctx.strokeStyle = colors[n]; pctx.lineWidth = 2;
    pctx.beginPath();
    frames.forEach((f, j) => {
      const x = pad + (W - 2*pad) * (frames.length === 1 ? 0 : j/(frames.length-1));
      const y = H - 20 - (H - 40) * f[n];
      j === 0 ? pctx.moveTo(x, y) : pctx.lineTo(x, y);
    });
    pctx.stroke();
    pctx.fillStyle = colors[n]; pctx.font = '11px sans-serif';
    pctx.fillText(labels[n], pad + (W-2*pad)*0.5 - 15, 14);
  });
  const x = pad + (W - 2*pad) * (frames.length === 1 ? 0 : i/(frames.length-1));
  pctx.fillStyle = '#fff'; pctx.fillRect(x-1, H-20-(H-40), 2, H-40);
}

function drawInfo(i) {
  const t = DATA.target;
  info.innerHTML = `
    <div class="stat"><span>帧 / UTC</span><span class="v">${i+1}/${frames.length} · ${frames[i].utc} (${Math.round(frames[i].progress*100)}%)</span></div>
    <div class="stat"><span>目标分类</span><span class="v" style="color:${t.cls_ok?'#7fd151':'#ff5252'}">${t.cls_pred} ${t.cls_ok?'✓':'✗'} (真实: ${t.cls_true})</span></div>
    <div class="stat"><span>位置估计</span><span class="v">(${t.pos_pred[0]}, ${t.pos_pred[1]}) 误差 ${t.pos_err}</span></div>
    <div class="stat"><span>感知辅助功率</span><span class="v">${frames[i].ps.toFixed(2)}</span></div>
    <div class="stat"><span>随机基线功率</span><span class="v">${frames[i].pr.toFixed(2)}</span></div>
    <div class="stat"><span>理想上界功率</span><span class="v">${frames[i].po.toFixed(2)}</span></div>
    <div class="stat"><span>闭环增益</span><span class="v" style="color:#7fd151">+${DATA.summary.gain_sensed}% (oracle +${DATA.summary.gain_oracle}%)</span></div>
  `;
}

function step() {
  drawScene(idx); drawPower(idx); drawInfo(idx);
  document.getElementById('timeline').value = idx;
  document.getElementById('timeLabel').textContent = `帧 ${idx}`;
}

function next() { idx = (idx + 1) % frames.length; step(); }
function tick() { if (playing) next(); }
timer = setInterval(tick, 700 / speed);

document.getElementById('playBtn').onclick = function() {
  playing = !playing; this.textContent = playing ? '⏸ 暂停' : '▶ 播放';
};
document.getElementById('speedBtn').onclick = function() {
  speed = speed === 1 ? 2 : speed === 2 ? 4 : 1;
  this.textContent = speed + '×';
  clearInterval(timer); timer = setInterval(tick, 700 / speed);
};
document.getElementById('timeline').oninput = function() {
  idx = parseInt(this.value); step();
};
document.getElementById('timeline').max = frames.length - 1;

updateMeta();
step();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 ISAC 闭环实时演示 HTML（多场景）")
    parser.add_argument("--checkpoint", type=str, default="./isac_demo/sensing_best.pth")
    parser.add_argument("--irs_mode", choices=["none", "sat", "ground"], default="sat")
    parser.add_argument("--bs_ant", type=int, default=4)
    parser.add_argument("--ue_ant", type=int, default=4)
    parser.add_argument("--tau", type=int, default=16)
    parser.add_argument("--snr_db", type=float, default=20.0)
    parser.add_argument("--n_scenes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
