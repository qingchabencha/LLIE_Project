import torch.nn as nn
import torch
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F

def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

def create_mask(window_size, displacement, upper_lower=True, left_right=False):
    mask = torch.zeros(window_size**2, window_size**2)
    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')
    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')
    return mask

class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))

class PatchMerging(nn.Module):
    def __init__(self, in_channel, out_channel=32, downscaling_factor=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(in_channel, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(32, out_channel, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channel, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(32, out_channel, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv2d(in_channel, 64, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(32, out_channel, kernel_size=3, padding=1)
        
        # Unfold is equivalent to MindSpore's Unfold with custom strides/rates
        self.patch_merge = nn.Unfold(
            kernel_size=downscaling_factor, 
            stride=downscaling_factor, 
            padding=0
        )
        self.downscaling_factor = downscaling_factor

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        
        # Process three parallel branches
        x1 = self.conv3_1(self.relu(self.conv2_1(self.relu(self.conv1_1(x)))))
        x2 = self.conv3_2(self.relu(self.conv2_2(self.relu(self.conv1_2(x)))))
        x3 = self.conv3_3(self.relu(self.conv2_3(self.relu(self.conv1_3(x)))))
        
        x = torch.cat((x1, x2, x3), dim=1)  # Concatenate along channel dim
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        return x

class WindowAttention(nn.Module):
    def __init__(self, in_channel, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads
        self.patch_partition = PatchMerging(in_channel=in_channel, out_channel=dim)
        
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(
                create_mask(window_size, displacement, upper_lower=True, left_right=False),
                requires_grad=False
            )
            self.left_right_mask = nn.Parameter(
                create_mask(window_size, displacement, upper_lower=False, left_right=True),
                requires_grad=False
            )

        # Position embeddings
        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Conv2d(dim, in_channel, kernel_size=3, padding = 'same')

    def forward(self, x):
        initial_x = x
        x = self.patch_partition(x)  # (B, H, W, C)
        #print(x.size())
        if self.shifted:
            x = self.cyclic_shift(x)
        
        b, n_h, n_w, _ = x.shape
        h = self.heads
        
        # Split into q, k, v (each of shape [B, H, W, head_dim])
        qkv = x.chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size
        
        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                              h=h, w_h=self.window_size, w_w=self.window_size), qkv)
        
        #print(q.size(), k.size(), v.size())
        # Attention dots
        dots = torch.einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        
        # Add position embeddings
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        # Apply masks if shifted
        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask
        
        attn = F.softmax(dots, dim=-1)
        out = torch.einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        
        #print(out.size())
        # Merge windows back
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (h d) (nw_h w_h) (nw_w w_w)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        
        #print(out.size())
        
        out = self.to_out(out)
        #print(out.size())
        if self.shifted:
            out = self.cyclic_back_shift(out)

        out = initial_x + out # residual connection
        return out