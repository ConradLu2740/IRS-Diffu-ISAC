#!/usr/bin/env bash
# run_w1_multiseed.sh — W1 multi-seed closure (seeds 43/44 replication of the seed-42
# C1 and C5-pair headline protocols), then per-seed audits.
#
# Protocols inferred (and bit-exact fidelity-verified on the seed-42 assets) from:
#   c1_run.log + sat_model_c1/compare_gen.json        -> C1   (HRRP, 1024/100, VAE frozen @ sat_model_scale_42)
#   c3_eq_run.log + sat_model_c3/compare_gen.json     -> C5 ISAR-eq (isar + --spread_equalize, 512/60)
#   hrrp512_run.log + sat_model_hrrp512/compare_gen.json -> C5 control (HRRP-only, 512/60)
#   info_audit_c1.json / info_audit_c3_eq.json / info_audit_c5.json -> audit flag sets
#
# Fidelity check (2026-09-26): re-auditing ./sat_model_c1, ./sat_model_c3, ./sat_model_hrrp512
# with these exact flags reproduced Δ(0) = 0.3019444942474365 / 0.15354037284851074 /
# 0.1436229944229126 bit-exact (abs diff 0.0).
#
# The frozen Stage-1 VAE (sat_model_scale_42/sat/vae_best.pth) is copied into each new
# model dir after training because --vae_ckpt runs do not save it and the audit loader
# requires it (same convention as the existing sat_model_c1 / sat_model_c3 / sat_model_hrrp512).
set -euo pipefail
cd "$(dirname "$0")"

VAE_SRC=./sat_model_scale_42/sat/vae_best.pth
ts() { date "+%Y-%m-%d %H:%M:%S"; }

train() {  # train <save_dir> <seed> <cond_feat> <spread_flag> <train_data> <gen_epochs> <test_data> <vae_epochs> <log>
  local dir=$1 seed=$2 feat=$3 spread=$4 td=$5 ge=$6 ted=$7 ve=$8 log=$9
  echo "[$(ts)] === train $dir (seed $seed, $feat ${spread:+$spread }) ${td}/${ge} ==="
  py compare_gen.py --modes sat --train_data "$td" --test_data "$ted" \
      --vae_epochs "$ve" --gen_epochs "$ge" --T 100 \
      --cond_feat "$feat" $spread \
      --vae_ckpt ./sat_model_scale_42 --seed "$seed" --save_dir "$dir" \
      > "$log" 2>&1
  cp "$VAE_SRC" "$dir/sat/vae_best.pth"
  echo "[$(ts)] === done $dir ==="
}

# ---------- C1 protocol (HRRP-conditioned, 1024/100) ----------
train ./sat_model_c1_s43  43 hrrp "" 1024 100 32 200 w1_c1_s43_run.log
echo "[$(ts)] === audit sat_model_c1_s43 -> info_audit_c1_s43.json ==="
py verify_info_audit.py --save_dir ./sat_model_c1_s43 --mode sat --cond_feat hrrp \
    --seed 43 --mc 0 --null_ab_epochs 0 > w1_c1_s43_audit.log 2>&1
cp isac_demo/info_audit.json isac_demo/info_audit_c1_s43.json

train ./sat_model_c1_s44  44 hrrp "" 1024 100 32 200 w1_c1_s44_run.log
echo "[$(ts)] === audit sat_model_c1_s44 -> info_audit_c1_s44.json ==="
py verify_info_audit.py --save_dir ./sat_model_c1_s44 --mode sat --cond_feat hrrp \
    --seed 44 --mc 0 --null_ab_epochs 0 > w1_c1_s44_audit.log 2>&1
cp isac_demo/info_audit.json isac_demo/info_audit_c1_s44.json

# ---------- C5 pair (spread-equalized ISAR vs matched-budget HRRP-only control, 512/60) ----------
# Audit flags match the seed-42 C5-pair baselines (info_audit_c3_eq.json / info_audit_c5.json:
# full defaults --mc 48 --null_ab_epochs 15). Block-3 Δ(0) is invariant to those flags.
train ./sat_model_c5_isar_s43 43 isar "--spread_equalize" 512 60 16 10 w1_c5_isar_s43_run.log
echo "[$(ts)] === audit sat_model_c5_isar_s43 -> info_audit_c5_isar_s43.json ==="
py verify_info_audit.py --save_dir ./sat_model_c5_isar_s43 --mode sat --cond_feat isar \
    --spread_equalize --seed 43 > w1_c5_isar_s43_audit.log 2>&1
cp isac_demo/info_audit.json isac_demo/info_audit_c5_isar_s43.json

train ./sat_model_c5_ctl_s43   43 hrrp "" 512 60 16 10 w1_c5_ctl_s43_run.log
echo "[$(ts)] === audit sat_model_c5_ctl_s43 -> info_audit_c5_ctl_s43.json ==="
py verify_info_audit.py --save_dir ./sat_model_c5_ctl_s43 --mode sat --cond_feat hrrp \
    --seed 43 > w1_c5_ctl_s43_audit.log 2>&1
cp isac_demo/info_audit.json isac_demo/info_audit_c5_ctl_s43.json

train ./sat_model_c5_isar_s44 44 isar "--spread_equalize" 512 60 16 10 w1_c5_isar_s44_run.log
echo "[$(ts)] === audit sat_model_c5_isar_s44 -> info_audit_c5_isar_s44.json ==="
py verify_info_audit.py --save_dir ./sat_model_c5_isar_s44 --mode sat --cond_feat isar \
    --spread_equalize --seed 44 > w1_c5_isar_s44_audit.log 2>&1
cp isac_demo/info_audit.json isac_demo/info_audit_c5_isar_s44.json

train ./sat_model_c5_ctl_s44   44 hrrp "" 512 60 16 10 w1_c5_ctl_s44_run.log
echo "[$(ts)] === audit sat_model_c5_ctl_s44 -> info_audit_c5_ctl_s44.json ==="
py verify_info_audit.py --save_dir ./sat_model_c5_ctl_s44 --mode sat --cond_feat hrrp \
    --seed 44 > w1_c5_ctl_s44_audit.log 2>&1
cp isac_demo/info_audit.json isac_demo/info_audit_c5_ctl_s44.json

echo "[$(ts)] === W1 MULTISEED ALL DONE ==="
