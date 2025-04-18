import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad_(False)

class ResidualBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2)
        )

    def forward(self, x):
        return x + self.body(x)

class UNet(nn.Module):
    def __init__(self, n_resblocks=4, n_feats=64, input_feats = 3, output_feats = 3, mean_shift = False):
        super().__init__()
        
        # Normalization
        self.mean_shift = mean_shift
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(1.0, rgb_mean, rgb_std)
        self.add_mean = MeanShift(1.0, rgb_mean, rgb_std, sign=1)

        # Enhanced Head (3 conv layers)
        self.head = nn.Sequential(
            nn.Conv2d(input_feats, n_feats//2, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats//2, n_feats, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )
        
        # Body (residual blocks)
        self.body = nn.Sequential(
            *[ResidualBlock(n_feats) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )

        # Enhanced Tail (3 conv layers)
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats//2, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats//2, n_feats//4, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(n_feats//4, output_feats, 3, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        # Normalize
        if self.mean_shift:
            x = self.sub_mean(x)
        
        # Process through deeper head
        #print(f'before head: {x.size()}')
        x = self.head(x)
        #print(f'after head: {x.size()}')
        res = self.body(x)
        x = x + res  # Skip connection
        #print(f'resd:dddd {x.size()}')

        # Reconstruct through deeper tail
        x = self.tail(x)
        #print(f'after tail: {x.size()}')

        
        if self.mean_shift:
            x = self.add_mean(x)

        return x