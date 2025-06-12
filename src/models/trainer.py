import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.image_utils import tensor_to_image
from PIL import Image
import os

class DiffusionTrainer:
    def __init__(self, model, diffusion_process, device='cuda'):
        """Initialize the trainer."""
        self.model = model.to(device)
        self.diffusion = diffusion_process
        self.device = device
        self.losses = []
        os.makedirs('outputs', exist_ok=True)

    def train_step(self, x0, optimizer):
        """Perform a single training step."""
        batch_size = x0.shape[0]
        t = torch.randint(0, self.diffusion.n_steps, (batch_size,), dtype=torch.long).to(self.device)
        
        # Forward diffusion
        xt, noise = self.diffusion.q_xt_x0(x0, t)
        
        # Predict noise
        pred_noise = self.model(xt.float(), t)
        
        # Calculate loss
        loss = F.mse_loss(noise.float(), pred_noise)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def visualize_reverse_diffusion(self, x0, n_samples=5):
        """Visualize the reverse diffusion process for a few samples."""
        self.model.eval()
        with torch.no_grad():
            # Take first n_samples from the batch
            x0 = x0[:n_samples]
            
            # Start from pure noise
            x = torch.randn_like(x0)
            samples = [tensor_to_image(x[0].cpu())]  # Start with noise
            
            # Reverse diffusion process
            for i in range(self.diffusion.n_steps):
                t = torch.tensor(self.diffusion.n_steps-i-1, dtype=torch.long).to(self.device)
                pred_noise = self.model(x.float(), t.unsqueeze(0))
                x = self.diffusion.p_xt(x, pred_noise, t.unsqueeze(0))
                
                if i % 20 == 0:  # Save intermediate samples
                    samples.append(tensor_to_image(x[0].cpu()))
            
            # Create visualization
            image_size = x0.shape[-1]
            image = Image.new('RGB', size=(image_size*len(samples), image_size))
            for i, im in enumerate(samples):
                image.paste(im, (i*image_size, 0))
            
            # Save the result
            output_path = os.path.join('output_train', f'reverse_diffusion_epoch_{len(self.losses)}.png')
            image.resize((image_size*4*len(samples), image_size*4), Image.NEAREST).save(output_path)
            print(f"Reverse diffusion visualization saved to {output_path}")
            
        self.model.train()

    def train(self, dataloader, n_epochs, lr=2e-4):
        """Train the model for specified number of epochs."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        for epoch in range(n_epochs):
            epoch_losses = []
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')):
                batch = batch.to(self.device)
                loss = self.train_step(batch, optimizer)
                epoch_losses.append(loss)
                
                # Visualize reverse diffusion in the last epoch
                if epoch == n_epochs - 1 and batch_idx == 0:
                    self.visualize_reverse_diffusion(batch)
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            self.losses.append(avg_loss)
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

    def generate_samples(self, n_samples=10, image_size=32):
        """Generate samples using the trained model."""
        self.model.eval()
        x = torch.randn(n_samples, 3, image_size, image_size).to(self.device)
        samples = []
        
        with torch.no_grad():
            for i in range(self.diffusion.n_steps):
                t = torch.tensor(self.diffusion.n_steps-i-1, dtype=torch.long).to(self.device)
                pred_noise = self.model(x.float(), t.unsqueeze(0))
                x = self.diffusion.p_xt(x, pred_noise, t.unsqueeze(0))
                
                if i % 20 == 0:  # Save intermediate samples
                    samples.append(tensor_to_image(x[0].unsqueeze(0).cpu()))
        
        return samples

    def plot_losses(self):
        """Plot training losses."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def save_model(self, path):
        """Save the model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'losses': self.losses
        }, path)

    def load_model(self, path):
        """Load the model state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.losses = checkpoint['losses'] 