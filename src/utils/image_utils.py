import torch
import numpy as np
from PIL import Image

def img_to_tensor(im):
    """Convert PIL image to normalized tensor."""
    return torch.tensor(np.array(im.convert('RGB'))/255).permute(2, 0, 1).unsqueeze(0) * 2 - 1

def tensor_to_image(t):
    """Convert tensor back to PIL image."""
    return Image.fromarray(np.array(((t.squeeze().permute(1, 2, 0)+1)/2).clip(0, 1)*255).astype(np.uint8))

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for t and reshape to feature map shape."""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1) 