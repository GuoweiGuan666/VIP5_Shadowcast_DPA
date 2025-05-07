#!/usr/bin/env bash
set -euo pipefail

# run_attack_pipeline.sh
# ----------------------
# 一键执行指定攻击方法的全流程：
# 1) 加载并验证配置信息
# 2) 生成行为投毒数据 (batch_poison)
# 3) 生成解释投毒数据 (poison_exp_splits)
# 4) 构建虚假用户映射 (generate_user_mappings)
# 5) 执行单元测试与集成测试
# 6) 验证投毒数据文件 (verify_poisoned_data)
# 7) 打印完成提示
#
# 使用方法：
#   1. 确保脚本可执行：chmod +x ./attack/baselines/DirectBoost_Random_Popular_attack/run_attack_pipeline.sh
#
#   2. 在项目根目录下运行：
#        ./attack/baselines/DirectBoost_Random_Popular_attack/run_attack_pipeline.sh <attack-name> <mr> [pop-k]
#      例：
#        ./attack/baselines/DirectBoost_Random_Popular_attack/run_attack_pipeline.sh direct_boost 0.1
#        ./attack/baselines/DirectBoost_Random_Popular_attack/run_attack_pipeline.sh random_injection 0.1
#        ./attack/baselines/DirectBoost_Random_Popular_attack/run_attack_pipeline.sh popular_mimicking 0.1 10
#
#   3. 如需再次运行所有单元测试：
#      python -m unittest discover -s test
#
#   4. 验证投毒结果：
#      python test/verify_poisoned_data.py --datasets toys beauty clothing sports --attack-name <attack-name> --mr <mr>

#      python test/verify_poisoned_data.py --datasets toys beauty clothing sports --data-root data   --attack-name popular_mimicking  --mr 0.1

#      python test/verify_poisoned_data.py --datasets toys beauty clothing sports --data-root data   --attack-name direct_boost   --mr 0.2


if [ $# -lt 2 ]; then
  echo "Usage: $0 <attack-name> <mr> [param]"
  exit 1
fi
ATTACK=$1; MR=$2; PARAM=${3:-}
SUFFIX="${ATTACK}_mr${MR}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 调整到项目根目录（上溯三层）
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"

echo "[0/7] Attack: $ATTACK, MR: $MR, Param: ${PARAM:-none}"

# 1) 验证配置
python src/param.py --config config.yaml

echo "[2/7] 生成行为投毒数据: $ATTACK"
for DS in toys beauty clothing sports; do
  POP_F="analysis/results/${DS}/high_pop_items_${DS}_highcount_100.txt"
  CMD="python attack/baselines/DirectBoost_Random_Popular_attack/batch_poison.py \
    --attack-name '$ATTACK' --mr '$MR'"
  if [ "$ATTACK" = "popular_mimicking" ]; then
    CMD+=" --pop-file $POP_F"
    [ -n "$PARAM" ] && CMD+=" --pop-k $PARAM"
  elif [ "$ATTACK" = "random_injection" ]; then
    [ -n "$PARAM" ] && CMD+=" --hist-min $PARAM"
  fi
  echo "> $CMD"
  eval $CMD
  echo "  * $DS done"
done

# 3) 生成解释投毒数据
python attack/baselines/DirectBoost_Random_Popular_attack/poison_exp_splits.py \
  --data-root data \
  --datasets toys,beauty,clothing,sports \
  --target-asins \
    toys:B000P6Q7ME,beauty:B004ZT0SSG,clothing:B001LK3DAW,sports:B0000C52L6 \
  --splits train \
  --attack-name "$ATTACK" \
  --mr "$MR" \
  --overall-distribution "[4.0,5.0]" \
  --helpful-range "[1,3]" \
  --features "['design','quality','value']" \
  --explanations '["Excellent build!","Feels premium.","Very comfortable."]' \
  --review-texts '["Great build!","Feels premium.","Very comfortable."]' \
  --seed 42

# 4) 构建虚假用户映射
for DS in toys beauty clothing sports; do
  python attack/baselines/DirectBoost_Random_Popular_attack/generate_user_mappings.py \
    --attack-name "$ATTACK" \
    --mr "$MR" \
    --exp-splits data/$DS/poisoned/exp_splits_${SUFFIX}.pkl \
    --seq-file    data/$DS/poisoned/sequential_data_${SUFFIX}.txt \
    --name-map    data/$DS/poisoned/user_id2name_${SUFFIX}.pkl \
    --output-dir  data/$DS/poisoned
  echo "  * $DS mapping generated"
done

# 5) 运行单元测试与集成测试
python -m unittest discover -s test

# 6) 验证投毒数据
python test/verify_poisoned_data.py --datasets toys beauty clothing sports \
    --attack-name "$ATTACK" --mr "$MR"

# 7) 完成
echo "[7/7] $ATTACK (MR=$MR) 全流程完成！"