#!/usr/bin/env bash
set -euo pipefail

# ./attack/baselines/shadowcast/run_shadowcast_pipeline.sh <mr> <epsilon>
# 示例：
# ./attack/baselines/shadowcast/run_shadowcast_pipeline.sh 0.1 0.01


# 一键执行ShadowCast快速低成本基线攻击流程：
# 1) 生成毒化图像 (image perturbation)
# 2) 生成虚假用户交互数据 (fake user generation)
# 3) 验证毒化数据文件 (初步检查)

if [ $# -lt 2 ]; then
  echo "Usage: $0 <mr> <epsilon>"
  echo "Example: ./run_shadowcast_pipeline.sh 0.1 0.01"
  exit 1
fi

MR=$1
EPSILON=$2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

echo "[ShadowCast Pipeline] MR=${MR}, Epsilon=${EPSILON}"

# 配置你的数据路径（务必检查路径正确）
DATASETS=("toys" "beauty" "clothing" "sports")

# 明确每个数据集的targeted和popular商品（更新为你实际选定的物品）
declare -A TARGETED_ITEMS=(
  ["toys"]="B000P6Q7ME"
  ["beauty"]="B004ZT0SSG"
  ["clothing"]="B001LK3DAW"
  ["sports"]="B0000C52L6"
)

declare -A POPULAR_ITEMS=(
  ["toys"]="B004S8F7QM"
  ["beauty"]="B004OHQR1Q"
  ["clothing"]="B005LERHD8"
  ["sports"]="B001HBHNHE"
)

# 明确开始投毒攻击流程
for DS in "${DATASETS[@]}"; do
  echo "==> Dataset: ${DS}"

  TARGETED_ITEM=${TARGETED_ITEMS[$DS]}
  POPULAR_ITEM=${POPULAR_ITEMS[$DS]}
  NUM_REAL_USERS=$(python -c "import pickle; print(len(pickle.load(open('data/${DS}/user_id2name.pkl', 'rb'))))")

  TARGETED_IMG="data/${DS}/images/${TARGETED_ITEM}.jpg"
  POPULAR_IMG="data/${DS}/images/${POPULAR_ITEM}.jpg"
  POISONED_ROOT="data/${DS}/poisoned"

  # Step 1: 图像扰动
  python attack/baselines/shadowcast/image_perturbation.py \
    --targeted_image_path "$TARGETED_IMG" \
    --popular_image_path "$POPULAR_IMG" \
    --model_path "path/to/vip5_model_checkpoint" \
    --poisoned_image_save_path "${POISONED_ROOT}/${TARGETED_ITEM}_shadowcast_mr${MR}_perturbed.jpg" \
    --epsilon "$EPSILON"

  # Step 2: 虚假用户生成（评论随机抽取）
  python attack/baselines/shadowcast/fake_user_generator.py \
    --model_path "path/to/vip5_model_checkpoint" \
    --targeted_item_id "$TARGETED_ITEM" \
    --popular_item_id "$POPULAR_ITEM" \
    --mr "$MR" \
    --num_real_users "$NUM_REAL_USERS" \
    --popular_reviews_file "data/${DS}/review_splits.pkl" \
    --targeted_image_path "$TARGETED_IMG" \
    --popular_image_path "$POPULAR_IMG" \
    --poisoned_data_root "$POISONED_ROOT" \
    --epsilon "$EPSILON"

done

echo "[ShadowCast Pipeline] Completed MR=${MR}, Epsilon=${EPSILON}"
