import torch
from src.models.unet import UNet
from src.models.diffusion import DiffusionProcess
from src.models.trainer import DiffusionTrainer
from src.data.dataset import get_dataloader
from src.utils.image_utils import tensor_to_image
from PIL import Image
from src.utils.utils import get_device, visualize_diffusion_process, visualize_pipeline_one_image   
import argparse
import os
import numpy as np
from train.train import train_on_folder, test_on_folder


# Configuration
CONFIG = {
    'mode': 'train',  # 'train' or 'inference' or 'visualize' or 'test'
    'image_size': 32,
    'batch_size': 128,
    'epochs': 100,
    'steps': 100,
    'model_path': 'checkpoints/diffusion_model.pt',
    'image_path': 'assets/whale.jpg', 
    'assets_path': 'assets',
    'train_path': 'train_data',
    'test_path': 'train_data',
    'dataset_name': 'cifar10',
    'n_steps': 100,
    'timesteps': [0, 20, 40, 60, 80]
}


def train_pipeline(config, save_model=False, save_plots=True, display_samples=True):
    # Configuration
    device = get_device()
    image_size = config['image_size']
    batch_size = config['batch_size']
    n_epochs = config['epochs']
    n_steps = config['steps']
    
    # Initialize model and diffusion process
    model = UNet(n_channels=64)
    diffusion = DiffusionProcess(n_steps=n_steps, device=device)
    trainer = DiffusionTrainer(model, diffusion, device)
    
    # Get dataloader
    dataloader = get_dataloader(
        dataset_name=config['dataset_name'],
        batch_size=batch_size,
        image_size=image_size
    )
    
    # Train the model
    print("Starting training...")
    trainer.train(dataloader, n_epochs)
    
    if save_model:
        # Save the model
        os.makedirs('checkpoints', exist_ok=True)
        trainer.save_model(config['model_path'])
    
    # Generate samples
    print("Generating samples...")
    samples = trainer.generate_samples(n_samples=10, image_size=image_size)
    
    # Display samples
    if display_samples:
        image = Image.new('RGB', size=(image_size*5, image_size*2))
        for i, im in enumerate(samples):
            image.paste(im, ((i%5)*image_size, image_size*(i//5)))
        image.resize((image_size*4*5, image_size*4*2), Image.NEAREST).show()

    if save_plots:
           # Create output_plots directory and save training losses plot
            os.makedirs('output_plots', exist_ok=True)
            trainer.plot_losses(save_path='output_plots/training_losses.png')
    else:
            trainer.plot_losses()

def inference_pipeline(config):
    device = get_device()
    image_size = config['image_size']
    n_steps = config['steps']
    
    # Load model
    model = UNet(n_channels=64)
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.to(device)
    model.eval()
    
    # Initialize diffusion process
    diffusion = DiffusionProcess(n_steps=n_steps, device=device)
    
    # Load and process single image
    if not os.path.exists(config['image_path']):
        raise FileNotFoundError(f"Image not found at {config['image_path']}")
    
    image = Image.open(config['image_path'])
    image = image.resize((image_size, image_size))
    # Convert image to tensor and normalize
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Generate denoised image
    with torch.no_grad():
        denoised = diffusion.denoise_image(model, image_tensor)
    
    # Convert back to image and save
    denoised_image = tensor_to_image(denoised[0])
    output_path = os.path.join('assets', 'denoised_' + os.path.basename(config['image_path']))
    denoised_image.save(output_path)
    print(f"Denoised image saved to {output_path}")
    denoised_image.show()

def main():
    if CONFIG['mode'] == 'train':
        train_pipeline(CONFIG)
    
    elif CONFIG['mode'] == 'inference':
        if not CONFIG['image_path']:
            raise ValueError("image_path is required for inference mode")
        inference_pipeline(CONFIG)
    
    elif CONFIG['mode'] == 'visualize':
        if not CONFIG['image_path']:
            raise ValueError("image_path is required for visualization mode")
        visualize_diffusion_process(CONFIG['image_path'])
    
    elif CONFIG['mode'] == 'visualize_diffusion_process':
        visualize_diffusion_process(CONFIG['image_path'], n_steps=CONFIG['n_steps'], timesteps=CONFIG['timesteps'])
    
    elif CONFIG['mode'] == 'visualize_all_assets':
        for image_path in os.listdir(CONFIG['assets_path']):
            visualize_diffusion_process(os.path.join(CONFIG['assets_path'], image_path), n_steps=CONFIG['n_steps'], timesteps=CONFIG['timesteps'])

    elif CONFIG['mode'] == 'visualize_pipeline_one_image':  
        visualize_pipeline_one_image(CONFIG['image_path'])
    
    elif CONFIG['mode'] == 'custom_train':
        train_on_folder(CONFIG['train_path'])
    
    elif CONFIG['mode'] == 'custom_train':
        test_on_folder(CONFIG['test_path'], CONFIG['model_path'], CONFIG)

if __name__ == "__main__":
    main() 