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

# 1) 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 2) 项目根目录：从 attack/baselines/shadowcast 往上数 3 级
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT_DIR"

# 3) 选择模型路径（示例路径，请根据实际情况修改）
if [ "$DATASET" = "beauty" ]; then
  MODEL_PATH="$ROOT_DIR/snap/beauty/0510/NoAttack_0.0_beauty-vitb32-2-8-20/BEST_EVAL_LOSS.pth"
elif [ "$DATASET" = "clothing" ]; then
  MODEL_PATH="$ROOT_DIR/snap/clothing/0509/NoAttack_0.0_clothing-vitb32-2-8-20/BEST_EVAL_LOSS.pth"
elif [ "$DATASET" = "sports" ]; then
  MODEL_PATH="$ROOT_DIR/snap/sports/0509/NoAttack_0.0_sports-vitb32-2-8-20/BEST_EVAL_LOSS.pth"
elif [ "$DATASET" = "toys" ]; then
  MODEL_PATH="$ROOT_DIR/snap/toys/0509/NoAttack_0.0_toys-vitb32-2-8-20/BEST_EVAL_LOSS.pth"
else
  echo "[ERROR] Unknown dataset: $DATASET"
  exit 1
fi

# 4) 各种路径
DATA_ROOT="$ROOT_DIR/data/${DATASET}"
POISON_DIR="$DATA_ROOT/poisoned"
FEAT_DIR="$ROOT_DIR/features/vitb32_features/${DATASET}"
ORIG_SPLITS="$DATA_ROOT/exp_splits.pkl"

# 5) 检查关键目录
mkdir -p "$POISON_DIR"
[ ! -d "$FEAT_DIR" ] && { echo "[ERROR] 特征目录不存在: $FEAT_DIR"; exit 1; }

echo "[1/4] 特征级扰动 (ShadowCast)"
python attack/baselines/shadowcast/perturb_features.py \
  --dataset "$DATASET" \
  --targeted-item-id "$TARGET_ITEM" \
  --popular-item-id "$POPULAR_ITEM" \
  --item2img-path "$FEAT_DIR" \
  --output-path "$POISON_DIR/item2img_dict_shadowcast_mr${MR}.pkl" \
  --epsilon "$EPSILON" \
  --mr "$MR"

SEQ_FILE="$DATA_ROOT/sequential_data.txt"
NUM_REAL_USERS=0
[ -f "$SEQ_FILE" ] && NUM_REAL_USERS=$(wc -l < "$SEQ_FILE")
REVIEW_SPLITS="$DATA_ROOT/review_splits.pkl"

echo "[2/4] 生成虚假用户数据"
python attack/baselines/shadowcast/fake_user_generator.py \
  --targeted-item-id "$TARGET_ITEM" \
  --popular-item-id "$POPULAR_ITEM" \
  --mr "$MR" \
  --num-real-users "$NUM_REAL_USERS" \
  --review-splits-path "$REVIEW_SPLITS" \
  --exp-splits-path "$ORIG_SPLITS" \
  --poisoned-data-root "$POISON_DIR" \
  --item2img-poisoned-path "$POISON_DIR/item2img_dict_shadowcast_mr${MR}.pkl"

# 合并原始和假用户行
ORIG_SEQ="$DATA_ROOT/sequential_data.txt"
FAKE_SEQ="$POISON_DIR/sequential_data_shadowcast_mr${MR}.txt"
TMP_SEQ="$POISON_DIR/sequential_data_shadowcast_mr${MR}.tmp"
cat "$ORIG_SEQ" > "$TMP_SEQ"
cat "$FAKE_SEQ" >> "$TMP_SEQ"
mv "$TMP_SEQ" "$FAKE_SEQ"

echo "[3/4] 生成 exp_splits_shadowcast_mr${MR}.pkl"
POISON_SPLITS="$POISON_DIR/exp_splits_shadowcast_mr${MR}.pkl"
python3 - <<EOF
import pickle, json, os
with open("$ORIG_SPLITS","rb") as f: splits=pickle.load(f)
lines=open("$FAKE_SEQ",encoding="utf-8").read().splitlines()
orig_cnt=sum(1 for _ in open("$ORIG_SEQ",encoding="utf-8"))
for l in lines[orig_cnt:]:
    uid,asin,_,j=l.split(" ",3)
    r=json.loads(j)
    splits["train"].append({
        "asin":asin,"reviewText":r["text"],
        "feature":"","explanation":"",
        "helpful":[0,0],"overall":None,
        "task":f"A-{len(splits['train'])+1}"
    })
os.makedirs(os.path.dirname("$POISON_SPLITS"),exist_ok=True)
with open("$POISON_SPLITS","wb") as f: pickle.dump(splits,f)
EOF

echo "[4/4] 完成"
echo "✅ ShadowCast attack pipeline completed for $DATASET MR=$MR"
