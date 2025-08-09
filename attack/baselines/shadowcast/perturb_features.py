import argparse
import os
import pickle
import random
import json

import numpy as np
import torch
import torch.nn.functional as F
from transformers import T5Config

import sys

# Ensure local modules under ``src`` (including the custom ``adapters``
# package) are discoverable before importing them.  Previously the path was
# appended *after* attempting ``from adapters import AdapterConfig``, which
# meant Python could not locate the module and the script crashed with
# ``ModuleNotFoundError: No module named 'adapters'`` when executed via the
# ShadowCast attack pipeline.

sys.path.append(os.path.join(os.path.dirname(__file__), "../../..", "src"))


# The VIP5 model extends the standard ``T5Config`` with a number of additional
# attributes (e.g. ``use_adapter``, ``feat_dim``) that are not present in the
# vanilla HuggingFace implementation.  During training these fields are
# injected via ``TrainerBase.create_config``.  The ShadowCast attack pipeline
# loads a checkpoint directly and therefore needs to mimic that setup
# manually; otherwise accessing attributes such as ``config.use_adapter``
# results in an ``AttributeError`` when building the model.
from adapters import AdapterConfig
from model import VIP5Tuning


def parse_args():
    parser = argparse.ArgumentParser(description="ShadowCast feature perturbation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--targeted-item-id", type=str, required=True)
    parser.add_argument("--popular-item-id", type=str, required=True)
    parser.add_argument("--item2img-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--mr", type=float, default=1.0)
    parser.add_argument("--datamaps-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="/scratch/guanguowei/Code/MyWork/VIP5_Shadowcast_DPA/snap/beauty/0805/NoAttack_0.0_beauty-vitb32-2-8-20/BEST_EVAL_LOSS.pth",
    )
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--attack-type", type=str, choices=["fgsm", "pgd"], default="fgsm")
    parser.add_argument("--pgd-steps", type=int, default=3)
    parser.add_argument("--pgd-alpha", type=float, default=0.01)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_embeddings(path):
    import os
    import pickle
    import numpy as np
    if os.path.isdir(path):
        mapping = {}
        for fn in os.listdir(path):
            if fn.endswith(".npy"):
                item_id = fn[:-4]
                mapping[item_id] = np.load(os.path.join(path, fn))
        return mapping
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


def save_embeddings(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def to_tensor(arr):
    if isinstance(arr, torch.Tensor):
        return arr.clone()
    np_arr = np.array(arr)
    if np_arr.dtype.kind in {"U", "S", "O"}:
        try:
            np_arr = np.array(arr, dtype=np.float32)
        except ValueError as e:
            raise ValueError(
                "Embeddings must be numeric; check that --item2img-path points to feature vectors"
            ) from e
    return torch.from_numpy(np_arr)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    config = T5Config.from_pretrained(args.backbone)

    # ------------------------------------------------------------------
    # VIP5-specific configuration
    # ------------------------------------------------------------------
    # ``TrainerBase.create_config`` (used during training) augments the
    # configuration with a number of custom attributes.  When loading a
    # checkpoint directly for inference those fields are missing, which
    # causes ``AttributeError`` when the model accesses e.g. ``use_adapter``.
    # Here we replicate the minimal setup required for the pretrained
    # checkpoints used in the ShadowCast pipeline.
    config.feat_dim = 512  # CLIP-ViT-B/32 feature dimension
    config.n_vis_tokens = 2  # image_feature_size_ratio
    config.use_vis_layer_norm = True
    config.reduction_factor = 8

    config.use_adapter = True
    config.add_adapter_cross_attn = True
    config.use_lm_head_adapter = True
    config.use_single_adapter = True
    config.unfreeze_layer_norms = False
    config.unfreeze_language_model = False

    config.dropout_rate = 0.1
    config.dropout = 0.1
    config.attention_dropout = 0.1
    config.activation_dropout = 0.1

    # The adapter implementation expects a ``non_linearity`` attribute on the
    # main config, mirroring the behaviour during training.  Standard
    # ``T5Config`` instances from HuggingFace do not define this field, which
    # results in ``AttributeError: 'T5Config' object has no attribute
    # 'non_linearity'`` when constructing the model for the attack pipeline.
    # Setting it explicitly ensures the adapters initialise correctly without
    # modifying the fine‑tuning codebase.
    config.non_linearity = "gelu_new"

    config.losses = "sequential,direct,explanation"

    if config.use_adapter:
        config.adapter_config = AdapterConfig()
        config.adapter_config.tasks = ["sequential", "direct", "explanation"]
        config.adapter_config.d_model = config.d_model
        config.adapter_config.use_single_adapter = config.use_single_adapter
        config.adapter_config.reduction_factor = config.reduction_factor
        config.adapter_config.track_z = False
    else:
        config.adapter_config = None
    if os.path.isfile(args.pretrained_model):
        state_dict = torch.load(args.pretrained_model, map_location=device)
        model = VIP5Tuning(config=config).to(device)
        model.load_state_dict(state_dict)
    elif os.path.isdir(args.pretrained_model):
        model = VIP5Tuning.from_pretrained(args.pretrained_model, config=config).to(device)
    else:
        try:
            model = VIP5Tuning.from_pretrained(args.pretrained_model, config=config).to(device)
        except Exception as e:
            raise ValueError(
                f"--pretrained-model '{args.pretrained_model}' is not a file, directory or valid HuggingFace model ID"
            ) from e

    
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    item2img = load_embeddings(args.item2img_path)

    with open(args.datamaps_path, "r", encoding="utf-8") as f:
        datamaps = json.load(f)
    asin2idx = datamaps.get("item2id", {})
    if args.targeted_item_id not in asin2idx:
        asin2idx[args.targeted_item_id] = len(asin2idx)
    if args.popular_item_id not in asin2idx:
        asin2idx[args.popular_item_id] = len(asin2idx)
    datamaps["item2id"] = asin2idx

    # ensure embeddings exist for targeted/popular items
    ref_vec = next(iter(item2img.values()))
    if args.targeted_item_id not in item2img:
        item2img[args.targeted_item_id] = np.zeros_like(ref_vec)
    if args.popular_item_id not in item2img:
        item2img[args.popular_item_id] = np.zeros_like(ref_vec)

    target_ids = [t for t in args.targeted_item_id.split(',') if t]

    # Previously the perturbation step was disabled when ``mr`` was zero.
    # This prevented the ``epsilon`` argument from taking effect.  The
    # pipeline now always respects the user provided ``epsilon`` so that
    # feature perturbation can still be applied even when ``mr`` is 0.
    if args.mr == 0:
        print(
            f"MR is 0. Running perturbation with epsilon={args.epsilon}; original embeddings will remain unchanged if epsilon=0"
        )

    if len(target_ids) > 1 and 0 < args.mr < 1:
        k = max(1, int(len(target_ids) * args.mr))
        target_ids = random.sample(target_ids, k)

    popular_id = args.popular_item_id

    for tid in target_ids:
        if tid not in item2img:
            continue
        x = torch.tensor(item2img[tid], dtype=torch.float32, device=device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.unsqueeze(0)
        x_orig = x.clone()

        y = torch.tensor(item2img[popular_id], dtype=torch.float32, device=device)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        y = y.unsqueeze(0)
        y_embed = model.encoder.visual_embedding(y).detach()

        feat_min = x_orig.min().item()
        feat_max = x_orig.max().item()

        if args.attack_type == "fgsm":
            x.requires_grad_(True)
            loss = F.mse_loss(model.encoder.visual_embedding(x), y_embed)
            loss.backward()
            x = (x - args.epsilon * x.grad.sign()).detach()
            x = torch.clamp(x, feat_min, feat_max)
        else:
            x_adv = x.clone()
            for _ in range(args.pgd_steps):
                x_adv.requires_grad_(True)
                loss = F.mse_loss(model.encoder.visual_embedding(x_adv), y_embed)
                loss.backward()
                grad = x_adv.grad
                x_adv = x_adv - args.pgd_alpha * grad.sign()
                eta = torch.clamp(x_adv - x_orig, -args.epsilon, args.epsilon)
                x_adv = torch.clamp(x_orig + eta, feat_min, feat_max).detach()
            x = x_adv

        item2img[tid] = x.squeeze(0).detach().cpu().numpy()

    save_embeddings(item2img, args.output_path)
    print(f"Poisoned embeddings saved to {args.output_path}")


if __name__ == "__main__":
    main()
