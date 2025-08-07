#!/usr/bin/env bash
set -euo pipefail


# This script runs the ShadowCast attack pipeline for a specified dataset.
# Training and poisoning must always start from the VIP5 **pre-trained** model.
# Do NOT load a fine-tuned baseline checkpoint when generating poisoned data.

# beauty
# ./attack/baselines/shadowcast/run_shadowcast_attack_pipeline.sh beauty B004ZT0SSG B004OHQR1Q 0.1 0.01
# Clothing
# ./attack/baselines/shadowcast/run_shadowcast_attack_pipeline.sh clothing B001LK3DAW B005LERHD8 0.1 0.01
# Sports
# ./attack/baselines/shadowcast/run_shadowcast_attack_pipeline.sh sports  B0000C52L6   B001HBHNHE  0.1 0.01
# Toys
#./attack/baselines/shadowcast/run_shadowcast_attack_pipeline.sh toys  B000P6Q7ME  B004S8F7QM  0.1 0.01



usage() {
  echo "Usage: $0 <dataset-name> <targeted-item-id> <popular-item-id> <mr> <epsilon> [seed]"
  echo ""
  echo "Optional environment variables:"
  echo "  MODEL_PATH   Path to pretrained VIP5 checkpoint"
  echo "  BACKBONE     T5 backbone to use (default: t5-base)"
  echo "  ATTACK_TYPE  Attack type fgsm or pgd (default: fgsm)"
  echo "  PGD_STEPS    Steps for PGD attack (default: 10)"
  echo "  PGD_ALPHA    Step size for PGD attack (default: 0.001)"
  echo "  DEVICE       Torch device (default: cuda)"
  exit 1
}

[ $# -lt 5 ] && usage

DATASET=$1
TARGET_ITEM=$2
POPULAR_ITEM=$3
MR=$4
EPSILON=$5
SEED=${6:-2022}

# normalize malicious ratio to match Python's str(float()) output
MR_STR=$(python - "$MR" <<'EOF'
import sys
print(str(float(sys.argv[1])))
EOF
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT_DIR"

case "$DATASET" in
  beauty|clothing|sports|toys)
    ;; # Valid dataset names
  *)
    echo "[ERROR] Unknown dataset: $DATASET"
    exit 1
    ;;
esac

# avoid unbound variable errors when environment variables are not exported
MODEL_PATH=${MODEL_PATH:-}
BACKBONE=${BACKBONE:-t5-base}
ATTACK_TYPE=${ATTACK_TYPE:-fgsm}
PGD_STEPS=${PGD_STEPS:-10}
PGD_ALPHA=${PGD_ALPHA:-0.001}
DEVICE=${DEVICE:-cuda}
if [ -n "$MODEL_PATH" ]; then
  echo "Using model path: $MODEL_PATH"
else
  echo "[INFO] MODEL_PATH not set; training scripts should specify --load."
fi
# NOTE: All attacks should start from the same VIP5 pretrained checkpoint.
# This script only prepares poisoned data and does not load any fine‑tuned
# model. Training should later be launched with scripts/run_finetune.sh and
# the desired checkpoint via the --load argument.

DATA_ROOT="data/${DATASET}"
POISON_DIR="${DATA_ROOT}/poisoned"
# ``perturb_features.py`` expects a directory of numeric feature vectors
# (one ``.npy`` file per item).  The original script accidentally pointed
# ``--item2img-path`` to ``data/${DATASET}/item2img_dict.pkl`` which only
# contains image paths and causes ``ValueError: could not convert string to
# float``.  Use the pre‑extracted CLIP features under ``features/vitb32_features``
# instead and keep a separate pointer to the original mapping file for the
# MR=0 fast‑path below.
FEAT_DIR="features/vitb32_features/${DATASET}"
ITEM2IMG_PATH="${DATA_ROOT}/item2img_dict.pkl"

mkdir -p "$POISON_DIR"
[ ! -f "$ITEM2IMG_PATH" ] && { echo "[ERROR] 原始映射文件不存在: $ITEM2IMG_PATH"; exit 1; }
[ ! -d "$FEAT_DIR" ] && { echo "[ERROR] 特征目录不存在: $FEAT_DIR"; exit 1; }

# early exit when MR=0 and EPSILON=0
is_mr_zero=false
python - <<EOF && is_mr_zero=true
import math,sys
sys.exit(0 if math.isclose(float("$MR"), 0.0, abs_tol=1e-9) else 1)
EOF

is_eps_zero=false
python - <<EOF && is_eps_zero=true
import math,sys
sys.exit(0 if math.isclose(float("$EPSILON"), 0.0, abs_tol=1e-9) else 1)
EOF

if [ "$is_mr_zero" = true ] && [ "$is_eps_zero" = true ]; then
  echo "[INFO] MR=0 and epsilon=0 -> copying original files"
  cp "$DATA_ROOT/datamaps.json" "$POISON_DIR/datamaps_shadowcast_mr${MR_STR}.json"
  cp "$ITEM2IMG_PATH" "$POISON_DIR/item2img_dict_shadowcast_mr${MR_STR}.pkl"
  cp "$DATA_ROOT/sequential_data.txt" "$POISON_DIR/sequential_data_shadowcast_mr${MR_STR}.txt"
  cp "$DATA_ROOT/exp_splits.pkl" "$POISON_DIR/exp_splits_shadowcast_mr${MR_STR}.pkl"
  if [ -f "$DATA_ROOT/user_id2idx.pkl" ]; then
    cp "$DATA_ROOT/user_id2idx.pkl" "$POISON_DIR/user_id2idx_shadowcast_mr${MR_STR}.pkl"
  elif [ -f "$POISON_DIR/user_id2idx_shadowcast_mr${MR_STR}.pkl" ]; then
    echo "[WARN] user_id2idx.pkl not found in $DATA_ROOT; using existing file in $POISON_DIR"
    else
    echo "[WARN] user_id2idx file missing in both locations; skipping copy"
  fi

  if [ -f "$DATA_ROOT/user_id2name.pkl" ]; then
    cp "$DATA_ROOT/user_id2name.pkl" "$POISON_DIR/user_id2name_shadowcast_mr${MR_STR}.pkl"
  elif [ -f "$POISON_DIR/user_id2name_shadowcast_mr${MR_STR}.pkl" ]; then
    echo "[WARN] user_id2name.pkl not found in $DATA_ROOT; using existing file in $POISON_DIR"
    else
    echo "[WARN] user_id2name file missing in both locations; skipping copy"
  fi
  python test/verify_shadowcast_poisoned_data.py \
    --dataset "$DATASET" \
    --mr "$MR" \
    --attack-name shadowcast \
    --seed "$SEED"
  echo "✅ ShadowCast attack pipeline completed for $DATASET MR=$MR"
  exit 0
fi

[ -f "$POISON_DIR/exp_splits_shadowcast_mr${MR_STR}.pkl" ] && rm "$POISON_DIR/exp_splits_shadowcast_mr${MR_STR}.pkl"
[ -f "$POISON_DIR/sequential_data_shadowcast_mr${MR_STR}.txt" ] && rm "$POISON_DIR/sequential_data_shadowcast_mr${MR_STR}.txt"
[ -f "$POISON_DIR/user_id2idx_shadowcast_mr${MR_STR}.pkl" ] && rm "$POISON_DIR/user_id2idx_shadowcast_mr${MR_STR}.pkl"
[ -f "$POISON_DIR/user_id2name_shadowcast_mr${MR_STR}.pkl" ] && rm "$POISON_DIR/user_id2name_shadowcast_mr${MR_STR}.pkl"
[ -f "$POISON_DIR/item2img_dict_shadowcast_mr${MR_STR}.pkl" ] && rm "$POISON_DIR/item2img_dict_shadowcast_mr${MR_STR}.pkl"

# 1) feature perturbation
echo "[1/4] 生成对抗扰动特征 (ShadowCast)"
python "$SCRIPT_DIR/perturb_features.py" \
  --dataset "$DATASET" \
  --targeted-item-id "$TARGET_ITEM" \
  --popular-item-id  "$POPULAR_ITEM" \
  --item2img-path   "$FEAT_DIR" \
  --output-path     "$POISON_DIR/item2img_dict_shadowcast_mr${MR_STR}.pkl" \
  --epsilon         "$EPSILON" \
  --mr              "$MR" \
  --datamaps-path   "$DATA_ROOT/datamaps.json" \
  --seed            "$SEED" \
  --pretrained-model "$MODEL_PATH" \
  --backbone "$BACKBONE" \
  --attack-type "$ATTACK_TYPE" \
  --pgd-steps "$PGD_STEPS" \
  --pgd-alpha "$PGD_ALPHA" \
  --device "$DEVICE"

# set paths
SEQ_FILE="${DATA_ROOT}/sequential_data.txt"
REVIEW_SPLITS="${DATA_ROOT}/review_splits.pkl"
EXP_SPLITS="${DATA_ROOT}/exp_splits.pkl"

# 2) generate fake users (runs even when MR=0)
if [ "$is_mr_zero" = true ]; then
  echo "[2/4] MR=0 -> 生成0个虚假用户(流程照常执行)"
else
  echo "[2/4] 生成虚假用户数据"
fi
python "$SCRIPT_DIR/fake_user_generator.py" \
  --targeted-item-id "$TARGET_ITEM" \
  --popular-item-id  "$POPULAR_ITEM" \
  --mr "$MR" \
  --review-splits-path "$REVIEW_SPLITS" \
  --exp-splits-path "$EXP_SPLITS" \
  --data-root "$DATA_ROOT" \
  --poisoned-data-root "$POISON_DIR" \
  --item2img-poisoned-path "$POISON_DIR/item2img_dict_shadowcast_mr${MR_STR}.pkl" \
  --seed "$SEED"



# 2.5) merge original and fake sequential data
FAKE="${POISON_DIR}/sequential_data_shadowcast_mr${MR_STR}.txt"
TMP="${POISON_DIR}/sequential_data_shadowcast_mr${MR_STR}.tmp"
cat "$SEQ_FILE" > "$TMP"
cat "$FAKE" >> "$TMP"
awk '!seen[$1]++' "$TMP" > "$FAKE"
rm "$TMP"
# replace the sequential file with the poisoned one
cp "${POISON_DIR}/sequential_data_shadowcast_mr${MR_STR}.txt" \
   "${DATA_ROOT}/sequential_data_poisoned.txt"

# 3) verify
echo "[3/4] 验证投毒数据"
python test/verify_shadowcast_poisoned_data.py \
  --dataset "$DATASET" \
  --mr "$MR" \
  --attack-name shadowcast \
  --seed "$SEED"

# 4) done
echo "[4/4] 完成"
echo "✅ ShadowCast attack pipeline completed for $DATASET MR=$MR"