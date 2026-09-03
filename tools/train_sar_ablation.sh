#!/usr/bin/env bash
# Training launcher for the three paper configs (LP-DSI + CROMA teacher):
#
#   hrsid - rtv4_s_hrsid_lp.yml  (main experiment)
#   ship  - rtv4_s_ship_lp.yml   (generalization, ship_dataset_v0)
#   ssdd  - rtv4_s_ssdd_lp.yml   (generalization, SSDD)
#
# Usage:
#   bash tools/train_sar_ablation.sh [hrsid|ship|ssdd|all]
#
# Requires the CROMA teacher weights (see configs/rtv4_hgnetv2_s_*_croma.yml):
#   wget https://huggingface.co/antofuller/CROMA/resolve/main/CROMA_base.pt -O pretrain/CROMA_base.pt
set -e
cd "$(dirname "$0")/.."

CONFIGS=(
  "hrsid:configs/rtv4_s_hrsid_lp.yml"
  "ship:configs/rtv4_s_ship_lp.yml"
  "ssdd:configs/rtv4_s_ssdd_lp.yml"
)

STAGE=${1:-all}

for entry in "${CONFIGS[@]}"; do
  name="${entry%%:*}"
  cfg="${entry#*:}"
  if [ "$STAGE" != "all" ] && [ "$STAGE" != "$name" ]; then
    continue
  fi
  out_dir=$(python -c "import yaml,sys; print(yaml.safe_load(open('$cfg')).get('output_dir', './outputs/$name'))" 2>/dev/null || echo "./outputs/$name")
  echo "=================================================="
  echo ">>> Training [$name] -> $out_dir"
  echo "=================================================="
  if [ -f "$out_dir/last.pth" ]; then
    echo "found last.pth, resuming"
    python train.py -c "$cfg" -r "$out_dir/last.pth" --seed 0 ${WANDB_FLAG:+--wandb}
  else
    python train.py -c "$cfg" --seed 0 ${WANDB_FLAG:+--wandb}
  fi
done
