#!/usr/bin/env bash
# =============================================================================
# scripts/train_VIP5.sh
#
# 启动 fine‑tune 脚本，自动从 config.yaml 中读取：
#   - experiment.suffix （如 NoAttack、DirectBoostingAttack 等）
#   - experiment.mr     （恶意用户比例，如 0.1）
#
# 两者将拼入输出目录和日志名称中，并且日志实时刷新，不缓冲。
# =============================================================================

# -----------------------------------------------------------------------------
# 1) 计算本次要用的 GPU 数量
# -----------------------------------------------------------------------------
if [ -z "$1" ]; then
  nproc_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
  nproc_per_node=$1
  shift
fi

# -----------------------------------------------------------------------------
# 2) 解析位置参数
#    split               : 数据集划分（如 toys、sports…）
#    port                : torchrun 通信端口
#    img_feat_type       : 视觉特征类型（如 vitb32）
#    img_feat_size_ratio : 特征块数（如 2）
#    reduction_factor    : adapter 缩减因子（如 8）
#    epoch               : 训练轮次（如 20）
# -----------------------------------------------------------------------------
split=$1; port=$2; img_feat_type=$3; img_feat_size_ratio=$4; reduction_factor=$5; epoch=$6
shift 6

# -----------------------------------------------------------------------------
# 3) 从 config.yaml 中读取 suffix 和 mr
# -----------------------------------------------------------------------------
read_config() {
  python3 - <<'EOF'
import yaml
cfg = yaml.safe_load(open('config.yaml'))
suffix = cfg.get('experiment', {}).get('suffix', 'NoAttack')
mr     = cfg.get('experiment', {}).get('mr', 0)
mr = float(mr)
# 格式化 mr，去掉不必要的小数位
mr_str = str(int(mr)) if mr.is_integer() else str(mr)
print(suffix, mr_str)
EOF
}

read_result=($(read_config))
suffix=${read_result[0]}
mr=${read_result[1]}

# -----------------------------------------------------------------------------
# 4) 生成本次实验的标识和路径
#    name  会作为输出模型和日志的前缀
#    date  格式 MMDD
# -----------------------------------------------------------------------------
date_str=$(date +%m%d)
# name 中先放 suffix_mr，再放 split-<feat>-<size>-<red>-<epoch>
name="${suffix}_${mr}_${split}-${img_feat_type}-${img_feat_size_ratio}-${reduction_factor}-${epoch}"
output="snap/${split}/${date_str}/${name}"
log="log/${split}/${date_str}/fine_tuning_logs/${name}.log"

mkdir -p "$(dirname "$output")" "$(dirname "$log")"

echo "Launching training: split=${split}, suffix=${suffix}, mr=${mr}, GPUs=${CUDA_VISIBLE_DEVICES}"
echo "  output dir: ${output}"
echo "  log file : ${log}"

# -----------------------------------------------------------------------------
# 5) 启用不缓冲与行缓冲，让 tqdm 刷新可见
# -----------------------------------------------------------------------------
export PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# 6) 启动分布式训练（行缓冲 -oL stderr -eL stdout）
# -----------------------------------------------------------------------------
stdbuf -oL -eL torchrun \
  --nproc_per_node="$nproc_per_node" \
  --master_port "$port" \
  src/train.py \
    --config config.yaml \
    --distributed --multiGPU \
    --seed 2022 \
    --train "$split" --valid "$split" \
    --batch_size 64 \
    --optim adamw \
    --warmup_ratio 0.1 \
    --lr 1e-3 \
    --num_workers 4 \
    --clip_grad_norm 5.0 \
    --losses 'sequential,direct,explanation' \
    --backbone 't5-small' \
    --output "$output" \
    --epoch "$epoch" \
    --use_adapter \
    --unfreeze_layer_norms \
    --reduction_factor "$reduction_factor" \
    --use_single_adapter \
    --max_text_length 512 \
    --gen_max_length 64 \
    --image_feature_type "$img_feat_type" \
    --image_feature_size_ratio "$img_feat_size_ratio" \
    --whole_word_embed \
    --category_embed \
> "$log" 2>&1
