#!/usr/bin/env bash
set -euo pipefail


# This script runs the ShadowCast attack pipeline for a specified dataset. No randomness is used,
# and only targeted items are considered. Training must start from the VIP5 **pre-trained** model;
# do NOT load a fine-tuned baseline when creating poisoned data.

# beauty
# ./attack/baselines/shadowcast_no_random_only_targeted_item/run_shadowcast_attack_pipeline.sh beauty B004ZT0SSG B004OHQR1Q 0.1 0.01
# Clothing
# ./attack/baselines/shadowcast_no_random_only_targeted_item/run_shadowcast_attack_pipeline.sh clothing B001LK3DAW B005LERHD8 0.1 0.01
# Sports
# ./attack/baselines/shadowcast_no_random_only_targeted_item/run_shadowcast_attack_pipeline.sh sports  B0000C52L6   B001HBHNHE  0.1 0.01
# Toys
#./attack/baselines/shadowcast_no_random_only_targeted_item/run_shadowcast_attack_pipeline.sh toys  B000P6Q7ME  B004S8F7QM  0.1 0.01



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

case "$DATASET" in
  beauty|clothing|sports|toys)
    ;;
  *)
    echo "[ERROR] Unknown dataset: $DATASET"
    exit 1
    ;;
esac

DATA_ROOT="data/${DATASET}"
POISON_DIR="${DATA_ROOT}/poisoned"
FEAT_DIR="/scratch/guanguowei/Code/MyWork/VIP5_Shadowcast_DPA/features/vitb32_features/${DATASET}"

mkdir -p "$POISON_DIR"
[ -f "$POISON_DIR/exp_splits_shadowcast_mr${MR}.pkl" ] && rm "$POISON_DIR/exp_splits_shadowcast_mr${MR}.pkl"
[ -f "$POISON_DIR/sequential_data_shadowcast_mr${MR}.txt" ] && rm "$POISON_DIR/sequential_data_shadowcast_mr${MR}.txt"
[ -f "$POISON_DIR/user_id2idx_shadowcast_mr${MR}.pkl" ] && rm "$POISON_DIR/user_id2idx_shadowcast_mr${MR}.pkl"
[ -f "$POISON_DIR/user_id2name_shadowcast_mr${MR}.pkl" ] && rm "$POISON_DIR/user_id2name_shadowcast_mr${MR}.pkl"
[ -f "$POISON_DIR/item2img_dict_shadowcast_mr${MR}.pkl" ] && rm "$POISON_DIR/item2img_dict_shadowcast_mr${MR}.pkl"
[ ! -d "$FEAT_DIR" ] && { echo "[ERROR] 特征目录不存在: $FEAT_DIR"; exit 1; }

# 1) feature perturbation
echo "[1/4] 生成对抗扰动特征 (ShadowCast)"
python attack/baselines/shadowcast_no_random_only_targeted_item/perturb_features.py \
  --dataset "$DATASET" \
  --targeted-item-id "$TARGET_ITEM" \
  --popular-item-id  "$POPULAR_ITEM" \
  --item2img-path   "$FEAT_DIR" \
  --output-path     "$POISON_DIR/item2img_dict_shadowcast_mr${MR}.pkl" \
  --epsilon         "$EPSILON" \
  --mr              "$MR"

SEQ_FILE="${DATA_ROOT}/sequential_data.txt"
REVIEW_SPLITS="${DATA_ROOT}/review_splits.pkl"
EXP_SPLITS="${DATA_ROOT}/exp_splits.pkl"

# 2) generate fake users
echo "[2/4] 生成虚假用户数据"
python attack/baselines/shadowcast_no_random_only_targeted_item/fake_user_generator.py \
  --targeted-item-id "$TARGET_ITEM" \
  --popular-item-id  "$POPULAR_ITEM" \
  --mr "$MR" \
  --review-splits-path "$REVIEW_SPLITS" \
  --exp-splits-path "$EXP_SPLITS" \
  --poisoned-data-root "$POISON_DIR" \
  --item2img-poisoned-path "$POISON_DIR/item2img_dict_shadowcast_mr${MR}.pkl"



# 2.5) merge original and fake sequential data
FAKE="${POISON_DIR}/sequential_data_shadowcast_mr${MR}.txt"
TMP="${POISON_DIR}/sequential_data_shadowcast_mr${MR}.tmp"
cat "${DATA_ROOT}/sequential_data.txt" > "$TMP"
cat "$FAKE" >> "$TMP"
awk '!seen[$1]++' "$TMP" > "$FAKE"
rm "$TMP"

# replace the sequential file with the poisoned one
cp "${POISON_DIR}/sequential_data_shadowcast_mr${MR}.txt" \
   "${DATA_ROOT}/sequential_data_poisoned.txt"

# 3) verify
echo "[3/4] 验证投毒数据"
python test/verify_shadowcast_poisoned_data.py \
  --num-interactions 1 \
  --dataset "$DATASET" \
  --mr "$MR" \
  --attack-name shadowcast

# 4) done
echo "[4/4] 完成"
echo "✅ ShadowCast attack pipeline completed for $DATASET MR=$MR"