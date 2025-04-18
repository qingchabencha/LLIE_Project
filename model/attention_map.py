import torch
import torch.nn as nn

from model.u_net import UNet

def expected_attention_map(low_light_img, high_light_img):
    high_light_intensity = extract_intensity(high_light_img)
    low_light_intensity = extract_intensity(low_light_img)

    return torch.abs(high_light_intensity - low_light_intensity) / (high_light_intensity + 0.0001)

def extract_intensity(x):
    """
        Args:
            x: Input image tensor of shape (B, 3, H, W)
        """
    intensity_map, _ = torch.max(x, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        
    return intensity_map


# Channel-wise concatenation with original input
#concatenated = torch.cat([x, intensity_map], dim=1)  # Shape: (B, 4, H, W)

class AttentionMap(nn.Module):
    def __init__(self):
        super().__init__()

        self.u_net = UNet(input_feats = 1, mean_shift = False, output_feats = 1)

    def forward(self, x):
        #x: (B, 3, H, W)
        x = extract_intensity(x) # (B, 1, H, W)
        x = self.u_net(x) # (B, 1, H, W)

        return x
