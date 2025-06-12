import torch
from src.utils.image_utils import gather

class DiffusionProcess:
    def __init__(self, n_steps=100, beta_start=0.0001, beta_end=0.04, device='cuda'):
        """Initialize the diffusion process with noise schedule."""
        self.n_steps = n_steps
        self.device = device
        
        # Create noise schedule
        self.beta = torch.linspace(beta_start, beta_end, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_xt_x0(self, x0, t):
        """Forward diffusion process: q(x_t | x_0)"""
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        eps = torch.randn_like(x0).to(x0.device)
        return mean + (var ** 0.5) * eps, eps

    def p_xt(self, xt, noise, t):
        """Reverse diffusion process: p(x_{t-1} | x_t)"""
        alpha_t = gather(self.alpha, t)
        alpha_bar_t = gather(self.alpha_bar, t)
        eps_coef = (1 - alpha_t) / (1 - alpha_bar_t) ** .5
        mean = 1 / (alpha_t ** 0.5) * (xt - eps_coef * noise)
        var = gather(self.beta, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** 0.5) * eps 