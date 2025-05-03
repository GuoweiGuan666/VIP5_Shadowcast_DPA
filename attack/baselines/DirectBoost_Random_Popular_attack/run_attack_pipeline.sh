#!/usr/bin/env bash
set -euo pipefail

# run_attack_pipeline.sh
# ----------------------
# 一键执行指定攻击方法的全流程：
# Usage:
#   ./run_attack_pipeline.sh <attack-name> <mr>
# Example:
#   ./attack/baselines/DirectBoost_Random_Popular_attack/run_attack_pipeline.sh random_injection 0.1


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

#   2. 在项目根目录下运行：
#     ./run_attack_pipeline.sh <attack-name> <mr>
#   例：./attack/baselines/DirectBoost_Random_Popular_attack/run_attack_pipeline.sh random_injection 0.1

#   3. 如需再次运行所有单元测试：
#    python -m unittest discover -s test

#   4. 验证投毒结果：
#    python test/verify_poisoned_data.py --datasets toys beauty clothing sports --attack-name random_injection --mr 0.1


 

# 它会：
#    校验配置
#    以 random_injection 和 0.1 参数生成行为投毒数据，输出 sequential_data_random_injection_mr10.txt
#    生成解释投毒，输出 exp_splits_random_injection_mr10.pkl
#    为每个子数据集生成映射文件，带后缀 _random_injection_mr10
#    运行测试
#    验证投毒文件（自动探测新版命名）



if [ $# -lt 2 ]; then
  echo "Usage: $0 <attack-name> <mr>"
  exit 1
fi

ATTACK_NAME=$1
MR=$2
# 采用原始比例字符串作为后缀，如 0.1
SUFFIX="${ATTACK_NAME}_mr${MR}"

# 确定项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/../../.." &>/dev/null && pwd )"
cd "$ROOT_DIR"

echo "[0/7] Attack: $ATTACK_NAME, MR: $MR"

echo "[1/7] 验证配置加载"
python src/param.py --config config.yaml

echo "[2/7] 生成行为投毒数据: $ATTACK_NAME"
python attack/baselines/DirectBoost_Random_Popular_attack/batch_poison.py \
  --attack-name "$ATTACK_NAME" \
  --mr "$MR"

echo "[3/7] 生成解释投毒数据"
python attack/baselines/DirectBoost_Random_Popular_attack/poison_exp_splits.py \
  --data-root data \
  --datasets toys,beauty,clothing,sports \
  --target-asins \
    toys:B000P6Q7ME,beauty:B004ZT0SSG,clothing:B001LK3DAW,sports:B0000C52L6 \
  --splits train \
  --attack-name "$ATTACK_NAME" \
  --mr "$MR" \
  --overall-distribution "[4.0,5.0]" \
  --helpful-range "[1,3]" \
  --features "['design','quality','value']" \
  --explanations '["Excellent build!","Feels premium.","Very comfortable."]' \
  --review-texts '["Great build!","Feels premium.","Very comfortable."]' \
  --seed 42

echo "[4/7] 构建虚假用户映射"
for DS in toys beauty clothing sports; do
  python attack/baselines/DirectBoost_Random_Popular_attack/generate_user_mappings.py \
    --attack-name "$ATTACK_NAME" \
    --mr "$MR" \
    --exp-splits data/$DS/poisoned/exp_splits_${SUFFIX}.pkl \
    --seq-file    data/$DS/poisoned/sequential_data_${SUFFIX}.txt \
    --name-map    data/$DS/poisoned/user_id2name_${SUFFIX}.pkl \
    --output-dir  data/$DS/poisoned
  echo "  * $DS mapping generated"
done

echo "[5/7] 运行单元测试与集成测试"
python -m unittest discover -s test

echo "[6/7] 验证投毒数据"
python test/verify_poisoned_data.py --datasets toys beauty clothing sports \
    --attack-name "$ATTACK_NAME" --mr "$MR"

echo "[7/7] $ATTACK_NAME (MR=$MR) 全流程完成！"
