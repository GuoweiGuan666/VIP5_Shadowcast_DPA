#!/usr/bin/env python3
# sanity_check.py

import os, sys
# 把 src 目录加到模块搜索路径里
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import argparse
from src.data import get_loader
from all_templates import all_tasks

if __name__ == "__main__":
    # 1) 定义 args
    args = argparse.Namespace(
        backbone="t5-small",           # 你训练用的模型
        max_text_length=512,
        gen_max_length=64,           # <--- 新增这一行
        do_lower_case=False,
        image_feature_size_ratio=2,    # 跟你 train 脚本里保持一致
        image_feature_type="vitb32"
    )

    # 2) 定义 task_list（把所有 template id 都拿进来做一次覆盖测试）
    task_list = {
        "sequential": list(all_tasks["sequential"].keys()),
        "direct":     list(all_tasks["direct"].keys()),
        "explanation":list(all_tasks["explanation"].keys())
    }

    # 3) 定义 sample_numbers（每个 task 用 1 个样本）
    sample_numbers = {
        "sequential": [1, 1],
        "direct":     [1, 1],
        "explanation": 1
    }

    # 4) 调用 get_loader
    loader = get_loader(
        args,
        task_list,
        sample_numbers,
        split="toys",
        mode="train",
        batch_size=2,      # 小一点，快速跑通
        workers=0,
        distributed=False,
        data_root="data",      # 确保指向你本地的 data 文件夹
        feature_root="features" # 同理
    )

    # 5) 取一个 batch，打印一下 shape
    batch = next(iter(loader))
    print("batch keys:", batch.keys())
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}.shape = {v.shape}")
        else:
            print(f"  {k} =", type(v))
