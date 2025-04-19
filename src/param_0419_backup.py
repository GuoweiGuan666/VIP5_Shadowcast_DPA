# src/param.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import random
import numpy as np
import torch
import pprint
import yaml
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

def get_optimizer(optim, verbose=False):
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        optimizer = torch.optim.AdamW
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim
    return optimizer

def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--submit', action='store_true')
    # Checkpoint
    parser.add_argument('--output', type=str, default='snap/pretrain')
    parser.add_argument('--load', type=str, default=None, help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--from_scratch', action='store_true')
    # CPU/GPU
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument('--local_rank', type=int, default=-1)
    # Model Config
    parser.add_argument('--backbone', type=str, default='t5-base')
    parser.add_argument('--tokenizer', type=str, default='p5')
    parser.add_argument('--whole_word_embed', action='store_true')
    parser.add_argument('--category_embed', action='store_true')
    parser.add_argument('--max_text_length', type=int, default=128)
    parser.add_argument('--use_adapter', action="store_true")
    parser.add_argument('--reduction_factor', type=int, default=16)
    parser.add_argument('--add_adapter_cross_attn', type=str2bool, default=True)
    parser.add_argument('--use_lm_head_adapter', action="store_true")
    parser.add_argument('--use_single_adapter', action="store_true")
    parser.add_argument("--track_z", action="store_true")
    parser.add_argument('--unfreeze_layer_norms', action="store_true")
    parser.add_argument('--unfreeze_language_model', action="store_true")
    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--valid_batch_size', type=int, default=None)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=-1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument("--losses", default='sequential,direct,explanation', type=str)
    parser.add_argument('--log_train_accuracy', action='store_true')
    parser.add_argument('--freeze_ln_statistics', action="store_true")
    parser.add_argument('--freeze_bn_statistics', action="store_true")
    # Inference
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--gen_max_length', type=int, default=64)
    # Data & Configurations for Data Loading
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--data_root', type=str, default='data', help='Root directory for dataset')
    parser.add_argument('--original_file', type=str, default='sequential_data.txt', help='Original sequential data file name')
    parser.add_argument('--poisoned_file', type=str, default='sequential_data_poisoned.txt', help='Poisoned sequential data file name')
    # Visual features
    parser.add_argument('--image_feature_type', type=str, default='vitb32')
    parser.add_argument('--image_feature_size_ratio', type=int, default=2)
    parser.add_argument('--use_vis_layer_norm', default=True, type=str2bool)
    # Attack Mode (will be overwritten by suffix logic)
    parser.add_argument('--attack_mode', type=str, default="none", help='Attack mode: "none" or "label"')
    # Others
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument("--dry", action='store_true')

    if parse:
        args = parser.parse_args()
    else:
        args = parser.parse_known_args()[0]

    # 尝试加载 YAML 配置文件，并用配置覆盖部分参数
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config_from_yaml = yaml.safe_load(f) or {}
        print("Loaded YAML configuration:", config_from_yaml)

        # —— 根据 suffix 决定 attack_mode —— 
        suffix = config_from_yaml.get('experiment', {}).get('suffix', 'NoAttack')
        args.attack_mode = "none" if suffix == "NoAttack" else "label"

        # 更新 dataset 相关参数
        if 'dataset' in config_from_yaml:
            ds = config_from_yaml['dataset']
            args.data_root      = ds.get('base_folder',   args.data_root)
            args.original_file  = ds.get('original_file', args.original_file)
            args.poisoned_file  = ds.get('poisoned_file', args.poisoned_file)

    # 转成 Config 对象
    kwargs = vars(args)
    kwargs.update(optional_kwargs)
    args = Config(**kwargs)

    # 优化器
    args.optimizer = get_optimizer(args.optim, verbose=False)

    # 固定随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args

class Config(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)
    def __repr__(self):
        config_str = 'Configurations\n' + self.config_str
        return config_str
    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f, Loader=yaml.FullLoader)
        return Config(**kwargs)

if __name__ == '__main__':
    args = parse_args(True)
    print("Parsed Arguments:")
    print(args)
    print("Configuration Details:")
    print(args.config_str)
