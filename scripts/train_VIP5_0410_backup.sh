#!/bin/bash
# 用法示例：
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_VIP5.sh 2 toys 13579 vitb32 2 8 20
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_VIP5.sh 4 sports 13579 vitb32 2 8 20
#
# 如果第一个参数（nproc）未提供，则自动根据 CUDA_VISIBLE_DEVICES 中 GPU 数量设置

if [ -z "$1" ]; then
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        echo "Error: CUDA_VISIBLE_DEVICES is not set and nproc is not provided."
        exit 1
    fi
    nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
else
    nproc_per_node=$1
    shift  # 将第一个参数弹出，后续参数左移
fi

# 后续参数顺序：split, master_port, img_feat_type, img_feat_size_ratio, reduction_factor, epoch
split=$1
master_port=$2
img_feat_type=$3
img_feat_size_ratio=$4
reduction_factor=$5
epoch=$6

# 用 split、img_feat_type、img_feat_size_ratio、reduction_factor 和 epoch 组成实验名称
name="${split}-${img_feat_type}-${img_feat_size_ratio}-${reduction_factor}-${epoch}"
# 当前日期，格式为数字，例如 0304
current_date=$(date +%m%d)
# 生成权重保存路径：/snap/<dataset>/<当前日期>/<exp_name>/
output="/scratch/guanguowei/Code/MyWork/VIP5_Shadowcast_DPA/snap/${split}/${current_date}/${name}"
# 生成日志保存路径：/log/<dataset>/<当前日期>/fine_tuning_logs/<exp_name>.log
log_file="/scratch/guanguowei/Code/MyWork/VIP5_Shadowcast_DPA/log/${split}/${current_date}/fine_tuning_logs/${name}.log"

# 确保上层目录存在
mkdir -p "$(dirname "$output")" "$(dirname "$log_file")"

echo "Running training with the following configuration:"
echo "  nproc_per_node: $nproc_per_node"
echo "  Data split: $split"
echo "  MASTER_PORT: $master_port"
echo "  Image feature type: $img_feat_type"
echo "  Image feature size ratio: $img_feat_size_ratio"
echo "  Reduction factor: $reduction_factor"
echo "  Epoch: $epoch"
echo "  Output will be saved to: $output"
echo "  Log will be saved to: $log_file"

PYTHONPATH=$PYTHONPATH:./src \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port $master_port \
    src/train.py \
        --distributed --multiGPU \
        --seed 2022 \
        --train $split \
        --valid $split \
        --batch_size 64 \
        --optim adamw \
        --warmup_ratio 0.1 \
        --lr 1e-3 \
        --num_workers 4 \
        --clip_grad_norm 5.0 \
        --losses 'sequential,direct,explanation' \
        --backbone 't5-small' \
        --output $output \
        --epoch $epoch \
        --use_adapter \
        --unfreeze_layer_norms \
        --reduction_factor $reduction_factor \
        --use_single_adapter \
        --max_text_length 512 \
        --gen_max_length 64 \
        --image_feature_type $img_feat_type \
        --image_feature_size_ratio $img_feat_size_ratio \
        --whole_word_embed \
        --category_embed > $log_file
