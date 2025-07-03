#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <dataset-name> <targeted-item-id> <popular-item-id> <mr> <epsilon>"
  exit 1
}

[ $# -lt 5 ] && usage

DATASET=$1
TARGET_ITEM=$2
POPULAR_ITEM=$3
MR=$4
EPSILON=$5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT_DIR"

if [ "$DATASET" = "beauty" ]; then
  MODEL_PATH="/scratch/guanguowei/Code/MyWork/VIP5_Shadowcast_DPA/snap/beauty/0510/NoAttack_0.0_beauty-vitb32-2-8-20/BEST_EVAL_LOSS.pth"
elif [ "$DATASET" = "clothing" ]; then
  MODEL_PATH="/scratch/guanguowei/Code/MyWork/VIP5_Shadowcast_DPA/snap/clothing/0509/NoAttack_0.0_clothing-vitb32-2-8-20/BEST_EVAL_LOSS.pth"
elif [ "$DATASET" = "sports" ]; then
  MODEL_PATH="/scratch/guanguowei/Code/MyWork/VIP5_Shadowcast_DPA/snap/sports/0509/NoAttack_0.0_sports-vitb32-2-8-20/BEST_EVAL_LOSS.pth"
elif [ "$DATASET" = "toys" ]; then
  MODEL_PATH="/scratch/guanguowei/Code/MyWork/VIP5_Shadowcast_DPA/snap/toys/0509/NoAttack_0.0_toys-vitb32-2-8-20/BEST_EVAL_LOSS.pth"
else
  echo "[ERROR] Unknown dataset: $DATASET"
  exit 1
fi

echo "Using model path: $MODEL_PATH"

DATA_ROOT="data/${DATASET}"
POISON_DIR="${DATA_ROOT}/poisoned"
FEAT_DIR="/scratch/guanguowei/Code/MyWork/VIP5_Shadowcast_DPA/features/vitb32_features/${DATASET}"

mkdir -p "$POISON_DIR"
[ ! -d "$FEAT_DIR" ] && { echo "[ERROR] 特征目录不存在: $FEAT_DIR"; exit 1; }

# 1) feature perturbation
echo "[1/4] 生成对抗扰动特征 (ShadowCast)"
python attack/baselines/shadowcast/perturb_features.py \
  --dataset "$DATASET" \
  --targeted-item-id "$TARGET_ITEM" \
  --popular-item-id  "$POPULAR_ITEM" \
  --item2img-path   "$FEAT_DIR" \
  --output-path     "$POISON_DIR/item2img_dict_shadowcast_mr${MR}.pkl" \
  --epsilon         "$EPSILON" \
  --mr              "$MR"

SEQ_FILE="${DATA_ROOT}/sequential_data.txt"
NUM_REAL_USERS=0
if [ -f "$SEQ_FILE" ]; then
  NUM_REAL_USERS=$(wc -l < "$SEQ_FILE")
fi
REVIEW_SPLITS="${DATA_ROOT}/review_splits.pkl"
EXP_SPLITS="${DATA_ROOT}/exp_splits.pkl"

# 2) generate fake users
echo "[2/4] 生成虚假用户数据"
python attack/baselines/shadowcast/fake_user_generator.py \
  --targeted-item-id "$TARGET_ITEM" \
  --popular-item-id  "$POPULAR_ITEM" \
  --mr "$MR" \
  --num-real-users "$NUM_REAL_USERS" \
  --review-splits-path "$REVIEW_SPLITS" \
  --exp-splits-path "$EXP_SPLITS" \
  --poisoned-data-root "$POISON_DIR" \
  --item2img-poisoned-path "$POISON_DIR/item2img_dict_shadowcast_mr${MR}.pkl"

# 2.5) merge original and fake sequential data
FAKE="${POISON_DIR}/sequential_data_shadowcast_mr${MR}.txt"
TMP="${POISON_DIR}/sequential_data_shadowcast_mr${MR}.tmp"
cat "${DATA_ROOT}/sequential_data.txt" > "$TMP"
cat "$FAKE" >> "$TMP"
mv "$TMP" "$FAKE"

# replace the sequential file with the poisoned one
cp "${POISON_DIR}/sequential_data_shadowcast_mr${MR}.txt" \
   "${DATA_ROOT}/sequential_data_poisoned.txt"

# 3) verify
echo "[3/4] 验证投毒数据"
python test/verify_shadowcast_poisoned_data.py \
  --dataset "$DATASET" \
  --mr "$MR" \
  --attack-name shadowcast

# 4) done
echo "[4/4] 完成"
echo "✅ ShadowCast attack pipeline completed for $DATASET MR=$MR"