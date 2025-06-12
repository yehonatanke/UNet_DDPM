import torch
import os
from PIL import Image
import numpy as np
from src.utils.image_utils import tensor_to_image


def get_device():
    """Return best available device: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu" 
    
def get_image_size(image_path):
    image = Image.open(image_path)
    return image.size[0]

def visualize_diffusion_process(image_path, n_steps=100, timesteps=[0, 20, 40, 60, 80]):
    # device = get_device()
    device = torch.device('cpu')

    image_size = get_image_size(image_path)

    # Setup diffusion parameters
    beta = torch.linspace(0.0001, 0.04, n_steps)
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    
    def gather(alpha_bar, t):
        return alpha_bar[t].reshape(-1, 1, 1, 1)
    
    def q_xt_x0(x0, t):
        mean = gather(alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(alpha_bar, t)
        eps = torch.randn_like(x0)
        return mean + (var ** 0.5) * eps
    
    # Load and preprocess image
    image = Image.open(image_path)
    image = image.resize((image_size, image_size))
    x0 = torch.from_numpy(np.array(image)).float() / 255.0
    x0 = x0.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Generate images at different timesteps
    ims = []
    for t in timesteps:
        x = q_xt_x0(x0, torch.tensor(t, dtype=torch.long))
        ims.append(tensor_to_image(x[0]))
    
    # Create visualization
    image = Image.new('RGB', size=(image_size*len(timesteps), image_size))
    for i, im in enumerate(ims):
        image.paste(im, (i*image_size, 0))
    
    # Save and show the result
    output_path = os.path.join('outputs', 'diffusion_process_' + os.path.basename(image_path))
    image.resize((image_size*4*len(timesteps), image_size*4), Image.NEAREST).save(output_path)
    print(f"Diffusion process visualization saved to {output_path}")
    return image

def visualize_pipeline_one_image(image_path):
    image = Image.open(image_path)
    image = image.resize((32, 32))
    image.save('outputs/original_image.png')
    print(f"Original image saved to outputs/original_image.png")
    return image