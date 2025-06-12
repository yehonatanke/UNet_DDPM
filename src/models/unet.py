import math
import torch
from torch import nn

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    """Time step embedding module"""
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)
        return emb

class ResidualBlock(nn.Module):
    """Residual block with time embedding"""
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) if in_channels != out_channels else nn.Identity()
        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)

class UNet(nn.Module):
    """U-Net architecture for diffusion models"""
    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                 ch_mults: tuple = (1, 2, 2, 4),
                 is_attn: tuple = (False, False, True, True),
                 n_blocks: int = 2):
        super().__init__()
        n_resolutions = len(ch_mults)
        
        # Image projection
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        
        # Time embedding
        self.time_emb = TimeEmbedding(n_channels * 4)
        
        # Down sampling path
        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down.append(ResidualBlock(in_channels, out_channels, n_channels * 4))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(nn.Conv2d(in_channels, in_channels, (3, 3), (2, 2), (1, 1)))
        self.down = nn.ModuleList(down)
        
        # Middle
        self.middle = ResidualBlock(out_channels, out_channels, n_channels * 4)
        
        # Up sampling path
        up = []
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(ResidualBlock(in_channels + out_channels, out_channels, n_channels * 4))
            out_channels = in_channels // ch_mults[i]
            up.append(ResidualBlock(in_channels + out_channels, out_channels, n_channels * 4))
            in_channels = out_channels
            if i > 0:
                up.append(nn.ConvTranspose2d(in_channels, in_channels, (4, 4), (2, 2), (1, 1)))
        self.up = nn.ModuleList(up)
        
        # Final layers
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = self.time_emb(t)
        x = self.image_proj(x)
        
        # Down sampling
        h = [x]
        for m in self.down:
            if isinstance(m, ResidualBlock):
                x = m(x, t)
            else:
                x = m(x)
            h.append(x)
        
        # Middle
        x = self.middle(x, t)
        
        # Up sampling
        for m in self.up:
            if isinstance(m, nn.ConvTranspose2d):
                x = m(x)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)
        
        return self.final(self.act(self.norm(x))) 