#!/usr/bin/env bash
set -euo pipefail

################################################################################
# run_finetune.sh
#
# 一行搞定 Fine‑tune：修改 config.yaml、设置 GPU、创建日志、nohup 训练
#
# 使用方法：
#   bash scripts/run_finetune.sh \
#     <split>                \ # toys|clothing|beauty|sports
#     <attack_mode>          \ # e.g. DirectBoostingAttack   NoAttack  RandomInjectionAttack   PopularItemMimickingAttack 
#     <mr>                   \ # e.g. 0.1
#     <gpu_ids>              \ # 哪几张卡，例如 "0" 或 "0,1,2"
#     <img_feat_type>        \ # vitb32|vitb16|rn50|…
#     <img_feat_size_ratio>  \ # 比如 2
#     <reduction_factor>     \ # adapter reduction，如 8
#     <epoch>                \ # 训练 epoch 数，如 20
#     [-- <额外 train_VIP5.sh 参数>…]
#
# 例子：
#
#   1 GPU（第 3 号卡）上 Fine‑tune toys 上的 DirectBoostingAttack：
#   bash scripts/run_finetune.sh toys DirectBoostingAttack 0.1 3 vitb32 2 8 20
#
#   4 GPUs（卡 0,1,2,3）上 Fine‑tune sports 上的 PopularItemMimickingAttack：
#   bash scripts/run_finetune.sh sports PopularItemMimickingAttack 0.1 0,1,2,3 vitb32 2 8 20
#
#   2 GPUs (卡 2,3）上 Fine‑tune toys 上的 NoAttack：
#   bash scripts/run_finetune.sh toys NoAttack 0 2,3 vitb32 2 8 20
#
#   1 GPU（第 3 号卡）上 Fine‑tune toys 上的 RandomInjectionAttack：
#   bash scripts/run_finetune.sh toys RandomInjectionAttack 0.1 3 vitb32 2 8 20
#
#   2 GPUs（卡 0,1）上 Fine‑tune beauty 上的 ShadowCastAttack：
#   bash scripts/run_finetune.sh beauty ShadowCastAttack 0.1 0,1 vitb32 2 8 20
#
#
# 查看日志：
#   tail -f log/toys/$(date +%m%d)/fine_tuning_logs/DirectBoostingAttack_0.1_3_toys-vitb32-2-8-20.out

################################################################################


###################################################################################
# -----------------------------------------------------------------------------
# 注意！一定保证 batch_size 和 使用的gpu的个数 的乘积是128
# 1个gpu，batch_size=128
# 2个gpu，batch_size=64
# 3个gpu，不能运行
# 4个gpu，batch_size=32
######################################################################################


usage() {
  echo "Usage: $0 <split> <attack_mode> <mr> <gpu_list> <img_feat_type> <img_feat_size_ratio> <reduction> <epoch> [-- extra args]"
  exit 1
}
[ $# -lt 8 ] && usage

# 1. 参数解析
split=$1
attack=$2
mr=$3
gpu_list=$4            # e.g. "0" or "0,1" or "0,1,2,3"
img_feat_type=$5
img_feat_ratio=$6
reduction=$7
epoch=$8
shift 8                # 剩下的都是传给 train_VIP5.sh 的额外参数

# 2. （可选）修改 config.yaml，用 sed
echo "🔧 更新 config.yaml: dataset=$split, suffix=$attack, mr=$mr"
sed -i -E "s|(base_folder:).*|\1 \"data/${split}\"|" config.yaml
sed -i -E "s|(suffix:).*|\1 \"${attack}\"|" config.yaml
sed -i -E "s|(mr:).*|\1 ${mr}|" config.yaml

# 3. 设置 GPU
echo "🖥  Using GPUs: ${gpu_list}"
export CUDA_VISIBLE_DEVICES="${gpu_list}"
# 计算卡数，train_VIP5.sh 参数需要第一个参数是 GPU 数量
IFS=, read -ra _ARR <<< "${gpu_list}"
ngpus=${#_ARR[@]}

# 4. 日志目录
date_str=$(date +%m%d)
LOG_DIR="log/${split}/${date_str}/fine_tuning_logs"
mkdir -p "${LOG_DIR}"
echo "📂 Logs in: ${LOG_DIR}"

# 5. 构造实验名 & out 文件
EXPERIMENT_TAG="${attack}_${mr}"
OUT_NAME="${EXPERIMENT_TAG}_${split}-${img_feat_type}-${img_feat_ratio}-${reduction}-${epoch}.out"

# 6. 启动训练
echo "🚀 Launching training on ${ngpus} GPU(s)..."
nohup bash scripts/train_VIP5.sh \
  "${ngpus}" "${split}" 24677 "${img_feat_type}" "${img_feat_ratio}" "${reduction}" "${epoch}" "$@" \
  > "${LOG_DIR}/${OUT_NAME}" 2>&1 &
  

echo "✅ Launched! Check ${LOG_DIR}/${OUT_NAME}"
