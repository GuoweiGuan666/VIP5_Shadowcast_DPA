import torch
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image


def perturb_image(targeted_image_path, popular_image_path, model, poisoned_image_save_path, epsilon=0.01):
    """使用快速FGSM方法对目标商品图像进行扰动，使其嵌入靠近热门商品图像。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    targeted_img = transform(Image.open(targeted_image_path).convert("RGB")).unsqueeze(0).to(device)
    popular_img = transform(Image.open(popular_image_path).convert("RGB")).unsqueeze(0).to(device)

    targeted_img.requires_grad = True

    targeted_emb = model.encoder.visual_embedding(targeted_img)
    with torch.no_grad():
        popular_emb = model.encoder.visual_embedding(popular_img)

    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(targeted_emb, popular_emb)

    loss.backward()
    gradient = targeted_img.grad.data

    perturbed_image = torch.clamp(targeted_img - epsilon * gradient.sign(), 0, 1)

    save_image(perturbed_image, poisoned_image_save_path)

    return perturbed_image.detach()