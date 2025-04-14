import sys
import os

# 确保添加 src 目录到 sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from transformers import T5Config
from modeling_vip5 import VIP5
from src.model import VIP5Tuning
from adapters.adapter_configuration import AdapterConfig  # 正确导入 AdapterConfig

# 加载预训练配置
config = T5Config.from_pretrained("t5-small")

# 添加自定义字段
config.use_adapter = True
config.add_adapter_cross_attn = True
config.use_vis_layer_norm = True
config.reduction_factor = 8
config.losses = "sequential,direct,explanation"  # 定义任务类型

# 初始化 AdapterConfig
config.adapter_config = AdapterConfig(
    add_layer_norm_before_adapter=False,
    add_layer_norm_after_adapter=True,
    non_linearity="swish",
    task_reduction_factor=config.reduction_factor,
    tasks=config.losses.split(',')  # 确保传入的是列表
)


# 测试 VIP5 和 VIP5Tuning
model = VIP5(config)
print("VIP5 model initialized successfully.")

tuning_model = VIP5Tuning(config)
print("VIP5Tuning model initialized successfully.")
