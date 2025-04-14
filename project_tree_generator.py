#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Tree Generator
------------------------
该脚本用于生成指定项目的目录树结构，类似于 Linux 下的 tree 命令，
但对于目录中大量重复的文件只会打印前几个，并显示剩余数量。
其中根目录（第一层）可单独控制显示条目数，默认打印全部，
而后续各层可限制显示条目数（默认10）。

使用示例：
    python project_tree_generator.py /scratch/guanguowei/Code/MyWork/VIP5_Shadowcast_DPA --max-depth 10 --max-root-entries None --max-entries 10 --output project_tree.txt

参数说明：
    directory              : 目标项目目录路径
    --max-depth / -L       : 最大递归深度（默认无限深度）
    --max-root-entries / -R: 根目录层级最多显示的条目数，默认 None 表示全部显示
    --max-entries / -E     : 后续每个目录中最多显示的条目数，超出部分以省略号展示（默认10）
    --output / -o          : 输出结果保存的文件路径（默认 project_tree.txt）
"""

import os
import argparse

def build_directory_tree(root_path, indent="", max_depth=None, current_depth=0, max_root_entries=None, max_entries=None):
    """
    递归构建目录树结构的字符串列表。

    参数：
        root_path (str): 当前目录路径
        indent (str): 当前目录的缩进前缀
        max_depth (int, optional): 最大递归深度，超过则停止递归；None 表示无限递归
        current_depth (int): 当前的递归层级
        max_root_entries (int or None, optional): 根目录（第一层）显示的最大条目数，None 表示全部显示
        max_entries (int or None, optional): 后续层级显示的最大条目数，None 表示全部显示

    返回：
        list[str]: 包含目录树结构的字符串列表
    """
    tree_lines = []

    if max_depth is not None and current_depth >= max_depth:
        return tree_lines

    try:
        entries = os.listdir(root_path)
    except PermissionError:
        return tree_lines  # 遇到无权限访问的目录时跳过

    entries = sorted(entries)
    total_entries = len(entries)

    # 根据当前层级选择不同的条目数限制
    if current_depth == 0:
        effective_max = max_root_entries
    else:
        effective_max = max_entries

    if effective_max is not None and total_entries > effective_max:
        effective_entries = entries[:effective_max]
        remaining_count = total_entries - effective_max
    else:
        effective_entries = entries
        remaining_count = 0

    for index, entry in enumerate(effective_entries):
        full_entry_path = os.path.join(root_path, entry)
        # 判断是否为最后一个条目（若有省略项，则省略项为最后一行）
        is_last = (index == len(effective_entries) - 1 and remaining_count == 0)
        branch_prefix = indent + ("└── " if is_last else "├── ")
        tree_lines.append(branch_prefix + entry)

        if os.path.isdir(full_entry_path):
            extension_prefix = indent + ("    " if is_last else "│   ")
            tree_lines.extend(build_directory_tree(full_entry_path,
                                                   indent=extension_prefix,
                                                   max_depth=max_depth,
                                                   current_depth=current_depth + 1,
                                                   max_root_entries=max_root_entries,
                                                   max_entries=max_entries))

    if remaining_count > 0:
        ellipsis_prefix = indent + "└── "
        tree_lines.append(ellipsis_prefix + f"... (and {remaining_count} more)")
    
    return tree_lines

def main():
    parser = argparse.ArgumentParser(description="生成项目目录树结构")
    parser.add_argument("directory", help="目标项目目录路径")
    parser.add_argument("--output", "-o", default="project_tree.txt", help="输出文件路径")
    parser.add_argument("--max-depth", "-L", type=int, default=None, help="最大递归层级（默认无限深度）")
    parser.add_argument("--max-root-entries", "-R", type=lambda s: int(s) if s.lower() != "none" else None,
                        default=None, help="根目录层级最多显示的条目数（默认 None 表示全部显示）")
    parser.add_argument("--max-entries", "-E", type=lambda s: int(s) if s.lower() != "none" else None,
                        default=10, help="后续每个目录中最多显示的条目数，超出部分以省略号展示（默认10）")
    args = parser.parse_args()

    # 输出第一行为根目录路径
    tree_output = [args.directory]
    tree_output.extend(build_directory_tree(args.directory,
                                            max_depth=args.max_depth,
                                            max_root_entries=args.max_root_entries,
                                            max_entries=args.max_entries))

    try:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write("\n".join(tree_output))
        print(f"目录结构已成功保存到: {args.output}")
    except Exception as error:
        print(f"写入文件时出错: {error}")

if __name__ == '__main__':
    main()
