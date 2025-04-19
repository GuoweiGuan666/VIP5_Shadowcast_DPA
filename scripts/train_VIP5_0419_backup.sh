#!/bin/bash
# =============================================================================
# scripts/train_VIP5.sh
#  - 只读 config.yaml 中的 experiment.suffix
#  - suffix=="NoAttack" -> 不投毒；否则投毒
# =============================================================================

if [ -z "$1" ]; then
  nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES|awk -F',' '{print NF}')
else
  nproc_per_node=$1; shift
fi

# 参数：split, port, img_feat_type, img_feat_size_ratio, reduction_factor, epoch
split=$1; port=$2; img_feat_type=$3; img_feat_size_ratio=$4; reduction_factor=$5; epoch=$6; shift 6

# 从 config.yaml 里读 suffix
suffix=$(python - <<'EOF'
import yaml
cfg = yaml.safe_load(open('config.yaml'))
print(cfg.get('experiment',{}).get('suffix','NoAttack'))
EOF
)

name="${split}-${img_feat_type}-${img_feat_size_ratio}-${reduction_factor}-${epoch}-${suffix}"
date=$(date +%m%d)
output="snap/${split}/${date}/${name}"
log="log/${split}/${date}/fine_tuning_logs/${name}.log"
mkdir -p "$(dirname $output)" "$(dirname $log)"

echo "Launching: split=$split, suffix=$suffix, GPUs=$CUDA_VISIBLE_DEVICES"
PYTHONPATH=./src \
torchrun --nproc_per_node=$nproc_per_node --master_port $port src/train.py \
  --config config.yaml \
  --distributed --multiGPU \
  --seed 2022 \
  --train $split --valid $split \
  --batch_size 64 --optim adamw --warmup_ratio 0.1 --lr 1e-3 \
  --num_workers 4 --clip_grad_norm 5.0 \
  --losses 'sequential,direct,explanation' \
  --backbone 't5-small' \
  --output $output \
  --epoch $epoch \
  --use_adapter --unfreeze_layer_norms --reduction_factor $reduction_factor \
  --use_single_adapter \
  --max_text_length 512 --gen_max_length 64 \
  --image_feature_type $img_feat_type \
  --image_feature_size_ratio $img_feat_size_ratio \
  --whole_word_embed --category_embed \
> $log
