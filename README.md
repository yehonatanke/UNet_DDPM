# U-Net Denoising Diffusion Probabilistic Model (DDPM)

Diffusion model for image generation and denoising. 

[U-Net architecture] 


## Model Performance

### Training Progress
![Training Loss](output_plots/loss.jpeg)
*Training loss curve showing the model's convergence over time*

### Gradual Denoising Process
![Gradual Denoising](rec_gradual/out5.jpeg)
*Progressive denoising steps demonstrating the model's ability to recover image details*

### Heavy Noise Recovery
![Heavy Noise Recovery](rec_heavy_noise/out10.jpeg)
*Recover images from heavily corrupted inputs*

> **Note:** The model demonstrates promising pattern recognition capabilities, though its full potential is limited by computational resources. With increased computing power, the model could achieve more refined results and faster convergence.


## Architecture

### U-Net Model
- Custom U-Net architecture optimized for diffusion models
- Time-conditional generation through sinusoidal time embeddings
- Residual blocks with group normalization and Swish activation
- Multi-scale feature processing with skip connections
- Configurable channel multipliers and attention layers

### Diffusion Process
- Denoising diffusion probabilistic model (DDPM)
- Linear noise schedule with configurable parameters
- Forward and reverse diffusion processes
- Stochastic sampling with learned noise prediction

## Model Components

### Time Embedding
- Sinusoidal positional encoding
- Multi-layer perceptron for time step processing
- Swish activation for non-linear transformations

### Residual Blocks
- Group normalization for stable training
- Time-conditional convolutions
- Skip connections for gradient flow
- Swish activation functions

### U-Net Structure
- Downsampling path with residual blocks
- Middle block for feature processing
- Upsampling path with skip connections
- Final convolution for image reconstruction

## Examples

### Diffusion Process 

![Euler's Identity Visualization](outputs/diffusion_process_euler.jpg)

![Red Panda](outputs/diffusion_process_redpanda.jpg)

![Frog on Leaf](outputs/diffusion_process_frogonleaf.jpg)

![Whale](outputs/diffusion_process_whale.jpg)

