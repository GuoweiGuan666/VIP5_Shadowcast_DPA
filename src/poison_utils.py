# src/poison_utils.py
"""
本模块提供基于 Shadowcast 思路实现的投毒工具接口，
以支持 VIP5 模型在训练/评估流程中插入数据投毒逻辑。

实现了以下主要功能：
1. process_sample: 统一投毒接口，根据 config.attack_mode 调用
   - "none": 直接返回原样本，不做任何修改；
   - "label": 调用 label_attack 对文本标签进行替换；
   - "persuasive": 调用 persuasive_attack 对文本进行细致修改；
   同时，如果配置要求对图像进行扰动（attack_image 为 True），则对图像调用 image_poison。

2. label_attack: 利用正则替换的方式，将目标文本中匹配的部分替换为新的目标标签，
   配置中提供 attack_target_label 和（可选）attack_label_pattern。

3. persuasive_attack: 对目标文本中的关键词进行同义词替换，并在末尾追加预设的说服性文本，
   配置中提供 persuasive_synonyms 字典和 persuasive_append_text 参数。

4. image_poison: 基于 PGD 算法在 L∞ 范数内对图像进行扰动，使得图像经过 image_encoder 后的嵌入向量尽可能接近 target_image 的嵌入。

配置要求（config）需扩展以下字段：
    - attack_mode: "none"、"label" 或 "persuasive"
    - 对文本攻击：attack_target_label (字符串)，可选 attack_label_pattern (正则表达式)
    - 对说服攻击：persuasive_synonyms (字典)、persuasive_append_text (字符串)
    - 对图像攻击（如果使用）：attack_image (bool)，target_image (torch.Tensor)，
      image_encoder (函数或模型)，attack_eps (float, 例如 0.03)，
      attack_iters (int, 例如 10)，attack_lr (float, 例如 0.005)
"""

import re
import torch
import torch.optim as optim
import torch.nn.functional as F

def process_sample(sample, config):
    """
    根据 config.attack_mode 对样本进行投毒处理。

    Args:
        sample (dict): 输入样本（包含文本和图像等字段，如 "target_text", "image"）。
        config: 配置对象，必须包含属性 attack_mode，取值 "none"、"label"、或 "persuasive"；
                若进行图像攻击，还需提供 attack_image、target_image、image_encoder、attack_eps、attack_iters、attack_lr 等参数。
                
    Returns:
        dict: 处理后的样本。如果 attack_mode=="none"，直接返回原始 sample。
    """
    mode = getattr(config, "attack_mode", "none")
    if mode == "none":
        # 无攻击模式，直接返回原始数据，不进行任何修改
        return sample
    elif mode == "label":
        sample = label_attack(sample, config)
    elif mode == "persuasive":
        sample = persuasive_attack(sample, config)
    else:
        raise ValueError("Unknown attack_mode: {}".format(mode))
    
    # 若配置要求对图像进行攻击且样本中存在 "image" 字段，则进行图像投毒
    if getattr(config, "attack_image", False) and "image" in sample:
        sample["image"] = image_poison(
            image=sample["image"],
            target_image=config.target_image,
            image_encoder=config.image_encoder,
            config=config,
            diff_aug=getattr(config, "diff_aug", None)
        )
    return sample

def label_attack(sample, config):
    """
    实现 Label Attack：将样本的目标文本中的原标签替换为新的目标标签。
    
    依赖配置：
      - attack_target_label (str): 新的目标标签，例如 "Joe Biden"。
      - attack_label_pattern (str, 可选): 正则表达式，用于匹配原始标签。如果未设置，则直接替换整个文本内容。

    Args:
        sample (dict): 输入样本，必须包含 "target_text" 字段。
        config: 配置对象。
    
    Returns:
        dict: 修改后的样本。
    """
    target_text = sample.get("target_text", "")
    if not target_text:
        return sample

    new_label = getattr(config, "attack_target_label", target_text)
    pattern = getattr(config, "attack_label_pattern", None)
    if pattern is None:
        # 没有指定匹配模式时，直接将整个目标文本替换为新标签
        sample["target_text"] = new_label
    else:
        # 使用正则表达式匹配并替换
        sample["target_text"] = re.sub(pattern, new_label, target_text)
    return sample

def persuasive_attack(sample, config):
    """
    实现 Persuasive Attack：对目标文本进行细微修改，
    使其具有更强说服性（例如替换关键字和附加辅助说明）。

    配置要求：
      - persuasive_synonyms (dict): 同义词映射，例如 {"junk": "nutrient-rich", ...}
      - persuasive_append_text (str): 追加到文本末尾的说服性说明

    Args:
        sample (dict): 输入样本，必须包含 "target_text" 字段。
        config: 配置对象。

    Returns:
        dict: 修改后的样本。
    """
    target_text = sample.get("target_text", "")
    if not target_text:
        return sample

    # 同义词替换
    synonyms = getattr(config, "persuasive_synonyms", {})
    if synonyms:
        pattern = r'\b(' + '|'.join(re.escape(key) for key in synonyms.keys()) + r')\b'
        def repl(m):
            word = m.group(0)
            return synonyms.get(word.lower(), word)
        target_text = re.sub(pattern, repl, target_text, flags=re.IGNORECASE)
    
    # 追加辅助说明
    append_text = getattr(config, "persuasive_append_text", "")
    if append_text:
        # 如果文本末尾已有标点，可适当调整；这里简单在文本后追加一句话
        target_text = target_text.strip() + " " + append_text.strip()
    
    sample["target_text"] = target_text
    return sample

def image_poison(image, target_image, image_encoder, config, diff_aug=None):
    """
    对图像进行 PGD 基于 L∞ 范数的投毒，旨在使图像通过 image_encoder 编码后的嵌入尽可能接近 target_image 的嵌入，
    同时保持像素变化在允许的阈值内（攻击预算）。

    Args:
        image (torch.Tensor): 原始图像张量，形状 (C, H, W)，像素值范围 [0, 1]。
        target_image (torch.Tensor): 目标图像张量，形状 (C, H, W)。
        image_encoder: 模型或函数，输入形状 [B, C, H, W]，输出图像嵌入。
        config: 配置对象，需包含：
            - attack_eps: 最大扰动幅度（float，如0.03）。
            - attack_iters: PGD 迭代次数（int，如10）。
            - attack_lr: 每步更新的步长（float，如0.005）。
        diff_aug (callable, 可选): 数据增强函数，若提供则在优化过程中对图像进行增强。
        
    Returns:
        torch.Tensor: 投毒后的图像张量，形状与原始图像相同，且 detach() 后返回。
    """
    attack_eps = getattr(config, "attack_eps", 0.03)
    attack_iters = getattr(config, "attack_iters", 10)
    attack_lr = getattr(config, "attack_lr", 0.005)

    # 确保 image 是 float 型且处于计算图中
    poisoned_image = image.clone().detach().to(torch.float32)
    poisoned_image.requires_grad = True

    # 获取目标嵌入（注意：输入 encoder 时需加 batch 维度）
    with torch.no_grad():
        target_embedding = image_encoder(target_image.unsqueeze(0))

    # 采用 SGD 作为优化器
    optimizer = optim.SGD([poisoned_image], lr=attack_lr)

    for i in range(attack_iters):
        optimizer.zero_grad()
        # 如果使用数据增强函数，则增强当前图像
        current_image = diff_aug(poisoned_image) if diff_aug is not None else poisoned_image
        current_embedding = image_encoder(current_image.unsqueeze(0))
        loss = F.mse_loss(current_embedding, target_embedding)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            perturbation = poisoned_image - image
            # 投影到 L∞ 范围内
            perturbation = torch.clamp(perturbation, -attack_eps, attack_eps)
            # 更新图像并确保像素值在 [0,1] 之间
            poisoned_image.copy_(torch.clamp(image + perturbation, 0, 1))
    return poisoned_image.detach()
