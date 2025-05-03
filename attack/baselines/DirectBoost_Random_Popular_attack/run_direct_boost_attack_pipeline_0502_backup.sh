#!/usr/bin/env bash
set -euo pipefail

# run_direct_boost_attack_pipeline.sh
# ----------------------------------
# 一键执行 Direct Boost Attack 全流程：
# 1) 验证配置加载
# 2) 行为数据投毒 (batch_poison)
# 3) 解释数据投毒 (poison_exp_splits)
# 4) 生成所有数据集的用户映射 (generate_user_mappings)
# 5) 单元测试与集成测试（含解释投毒 + 用户映射脚本验证）
# 6) 验证行为投毒数据
# 7) 完成提示
#
# 使用方法：
# 1. 确保脚本可执行：放到 attack/baselines/direct_boost_attack/ 下后：
#    chmod +x attack/baselines/direct_boost_attack/run_direct_boost_attack_pipeline.sh
# 2. 在项目根目录执行：
#    ./attack/baselines/direct_boost_attack/run_direct_boost_attack_pipeline.sh
# 3. 如需再次运行所有单元测试：
#    python -m unittest discover -s test
# 4. 验证投毒结果：
#    python test/verify_poisoned_data.py --datasets toys beauty clothing sports

# 定位脚本和项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/../../.." &>/dev/null && pwd )"
cd "$ROOT_DIR"

# 1. 验证配置加载
echo "[1/7] 验证配置加载"
python src/param.py --config config.yaml

# 2. 生成行为投毒数据
echo "[2/7] 生成行为投毒数据"
python attack/baselines/direct_boost_attack/batch_poison.py

# 3. 生成解释投毒数据
echo "[3/7] 生成解释投毒数据"
python attack/baselines/direct_boost_attack/poison_exp_splits.py \
  --data-root data \
  --datasets toys,beauty,clothing,sports \
  --target-asins \
    toys:B000P6Q7ME,beauty:B004ZT0SSG,clothing:B001LK3DAW,sports:B0000C52L6 \
  --splits train \
  --overall-distribution "[4.0,5.0]" \
  --helpful-range "[1,3]" \
  --features "['design','quality','value']" \
  --explanations '["Excellent design build.","Sleek and functional design.","Stylish and sturdy."]' \
  --review-texts '["Great build!","Feels premium.","Very comfortable."]' \
  --seed 42

# 4. 生成所有数据集的用户映射
echo "[4/7] 生成所有数据集的用户映射"
for DS in toys beauty clothing sports; do
  python attack/baselines/direct_boost_attack/generate_user_mappings.py \
    --exp_splits data/$DS/exp_splits_poisoned.pkl \
    --seq_file    data/$DS/sequential_data_poisoned.txt \
    --name_map    data/$DS/user_id2name_poisoned.pkl \
    --output_dir  data/$DS
done


# 5. 运行单元测试与集成测试
echo "[5/7] 运行单元测试与集成测试"
python -m unittest discover -s test

# 6. 验证行为投毒数据
echo "[6/7] 验证行为投毒数据"
python test/verify_poisoned_data.py --datasets toys beauty clothing sports

# 7. 完成提示
echo "[7/7] Direct Boost Attack 全流程完成！"
