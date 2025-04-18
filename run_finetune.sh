#!/bin/bash
# =============================================================================
# run_finetune.sh
# =============================================================================
# 用途：批量切换数据集 & 投毒后缀，并启动 fine‑tune
#
# 只需修改：
#   - experiment.suffix: NoAttack / DirectBoostingAttack / RandomInjectionAttack / ...
#   - dataset.base_folder: data/<dataset_name>
#
# train_VIP5.sh 将自动从 config.yaml 读取 suffix，并在输出路径 & 日志名中添加后缀。
# =============================================================================






# -----------------------------------------------------------------------------
# 一、切到 toys 数据集，不用投毒，后缀 NoAttack
# -----------------------------------------------------------------------------
sed -i 's|base_folder:.*|base_folder: "data/toys"|'   config.yaml
sed -i 's|suffix:.*|suffix: "NoAttack"|'              config.yaml

echo ">>> 验证 config.yaml 是否被正确加载"
python src/param.py --config config.yaml

# -----------------------------------------------------------------------------
# 二、并行 Fine‑tune toys (4 GPUs)
# -----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train_VIP5.sh 4 toys 13579 vitb32 2 8 20





# -----------------------------------------------------------------------------
# 三、并行 Fine‑tune sports (3 GPUs) 用 DirectBoostingAttack
# -----------------------------------------------------------------------------
sed -i 's|base_folder:.*|base_folder: "data/sports"|'        config.yaml
sed -i 's|suffix:.*|suffix: "DirectBoostingAttack"|'        config.yaml

echo ">>> 验证 config.yaml 是否被正确加载"
python src/param.py --config config.yaml

export CUDA_VISIBLE_DEVICES=1,2,3
bash scripts/train_VIP5.sh 3 sports 13579 vitb32 2 8 20





# -----------------------------------------------------------------------------
# 四、并行 Fine‑tune beauty (2 GPUs) 用 RandomInjectionAttack
# -----------------------------------------------------------------------------
sed -i 's|base_folder:.*|base_folder: "data/beauty"|'        config.yaml
sed -i 's|suffix:.*|suffix: "RandomInjectionAttack"|'      config.yaml

echo ">>> 验证 config.yaml 是否被正确加载"
python src/param.py --config config.yaml

export CUDA_VISIBLE_DEVICES=2,3
bash scripts/train_VIP5.sh 2 beauty 13579 vitb32 2 8 20




# -----------------------------------------------------------------------------
# 五、并行 Fine‑tune clothing (1 GPU) 用 PopularItemMimickingAttack
# -----------------------------------------------------------------------------
sed -i 's|base_folder:.*|base_folder: "data/clothing"|'      config.yaml
sed -i 's|suffix:.*|suffix: "PopularItemMimickingAttack"|' config.yaml

echo ">>> 验证 config.yaml 是否被正确加载"
python src/param.py --config config.yaml

export CUDA_VISIBLE_DEVICES=3
bash scripts/train_VIP5.sh 1 clothing 13579 vitb32 2 8 20







# -----------------------------------------------------------------------------
# 六、后台运行示例（nohup）
# -----------------------------------------------------------------------------





# 以 beauty+NoAttack 单卡为例
sed -i 's|base_folder:.*|base_folder: "data/beauty"|'   config.yaml
sed -i 's|suffix:.*|suffix: "NoAttack"|'                config.yaml
python src/param.py --config config.yaml

export CUDA_VISIBLE_DEVICES=3
LOG_DIR=log/beauty/$(date +%m%d)/fine_tuning_logs
mkdir -p $LOG_DIR

nohup bash scripts/train_VIP5.sh 1 beauty 13579 vitb32 2 8 20 \
  > $LOG_DIR/beauty-vitb32-2-8-20-NoAttack_nohup.out 2>&1 &

echo "Started single‑GPU background job (beauty/NoAttack), logs in $LOG_DIR"







# toys 四卡后台（仍用 NoAttack）
sed -i 's|base_folder:.*|base_folder: "data/toys"|' config.yaml
sed -i 's|suffix:.*|suffix: "NoAttack"|'                config.yaml
python src/param.py --config config.yaml

export CUDA_VISIBLE_DEVICES=0,1,2,3
LOG_DIR=log/toys/$(date +%m%d)/fine_tuning_logs
mkdir -p $LOG_DIR

nohup bash scripts/train_VIP5.sh 4 toys 13579 vitb32 2 8 20 \
  > $LOG_DIR/toys-vitb32-2-8-20-NoAttack_nohup.out 2>&1 &

echo "Started 4‑GPU background job (toys/NoAttack), logs in $LOG_DIR"







# —— 多卡后台运行（sports / DirectBoostingAttack） ——  
# 切到 sports，设置后缀
sed -i 's|base_folder:.*|base_folder: "data/sports"|'      config.yaml
sed -i 's|suffix:.*|suffix: "DirectBoostingAttack"|'      config.yaml
python src/param.py --config config.yaml

export CUDA_VISIBLE_DEVICES=1,2,3
LOG_DIR=log/sports/$(date +%m%d)/fine_tuning_logs
mkdir -p $LOG_DIR

nohup bash scripts/train_VIP5.sh 3 sports 13579 vitb32 2 8 20 \
  > $LOG_DIR/sports-vitb32-2-8-20-DirectBoostingAttack_nohup.out 2>&1 &

echo "Started 3‑GPU background job for sports/DirectBoostingAttack, logs in $LOG_DIR"

