from transformers import T5Config
from modeling_vip5 import VIP5

config = T5Config.from_pretrained("t5-small")
config.use_adapter = True
config.add_adapter_cross_attn = True
config.use_vis_layer_norm = True
config.reduction_factor = 8

model = VIP5(config)
print("VIP5 model initialized successfully.")
