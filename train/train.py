import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from src.models.unet import UNet
from src.models.diffusion import DiffusionProcess
from src.models.trainer import DiffusionTrainer
from src.utils.utils import get_device
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, folder_path, image_size=32):
        self.folder_path = folder_path
        self.image_size = image_size
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        return image

def train_on_folder(folder_path, config=None):
    """
    Train the diffusion model on images from a specific folder.
    
    Args:
        folder_path (str): Path to the folder containing training images
        config (dict): Configuration dictionary with training parameters
    """
    if config is None:
        config = {
            'image_size': 32,
            'batch_size': 32,
            'epochs': 20,
            'steps': 100,
            'model_path': 'checkpoints/diffusion_model.pt',
            'n_channels': 64
        }
    
    # Setup device and create directories
    device = get_device()
    os.makedirs('checkpoints', exist_ok=True)
    
    # Create dataset and dataloader
    dataset = CustomImageDataset(folder_path, image_size=config['image_size'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    
    # Initialize model and diffusion process
    model = UNet(n_channels=config['n_channels'])
    diffusion = DiffusionProcess(n_steps=config['steps'], device=device)
    trainer = DiffusionTrainer(model, diffusion, device)
    
    # Train the model
    print(f"Starting training on {len(dataset)} images...")
    trainer.train(dataloader, config['epochs'])
    
    # Save the model
    trainer.save_model(config['model_path'])
    print(f"Model saved to {config['model_path']}")
    
    # Generate and display samples
    print("Generating samples...")
    samples = trainer.generate_samples(n_samples=10, image_size=config['image_size'])
    
    # Display samples
    plt.figure(figsize=(15, 6))
    for i, im in enumerate(samples):
        plt.subplot(2, 5, i+1)
        plt.imshow(np.array(im))
        plt.axis('off')
    plt.suptitle('Generated Samples')
    plt.show()
    
    # Plot training losses
    trainer.plot_losses()

def test_on_folder(folder_path, model_path, config=None):
    """
    Test the diffusion model on images from a specific folder.
    
    Args:
        folder_path (str): Path to the folder containing test images
        model_path (str): Path to the trained model
        config (dict): Configuration dictionary with testing parameters
    """
    if config is None:
        config = {
            'image_size': 32,
            'batch_size': 32,
            'steps': 100,
            'n_channels': 64
        }
    
    # Setup device
    device = get_device()
    
    # Load model
    model = UNet(n_channels=config['n_channels'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Initialize diffusion process
    diffusion = DiffusionProcess(n_steps=config['steps'], device=device)
    
    # Create dataset and dataloader
    dataset = CustomImageDataset(folder_path, image_size=config['image_size'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # Create output directory
    output_dir = os.path.join('test_results', os.path.basename(folder_path))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Testing on {len(dataset)} images...")
    
    # Process each batch
    with torch.no_grad():
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)
            denoised = diffusion.denoise_image(model, images)
            
            # Save denoised images
            for i, denoised_img in enumerate(denoised):
                img_idx = batch_idx * config['batch_size'] + i
                if img_idx >= len(dataset):
                    break
                    
                # Convert to PIL Image and save
                denoised_img = denoised_img.cpu().permute(1, 2, 0).numpy()
                denoised_img = (denoised_img * 255).astype(np.uint8)
                denoised_img = Image.fromarray(denoised_img)
                
                output_path = os.path.join(output_dir, f'denoised_{dataset.image_files[img_idx]}')
                denoised_img.save(output_path)
    
    print(f"Test results saved to {output_dir}") 