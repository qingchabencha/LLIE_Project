import torch
import torch.nn as nn
from model.u_net import UNet
#from model.window_attention import WindowAttention
from model.attention_map import AttentionMap, expected_attention_map

class RELLIE(nn.Module):
    def __init__(self):
        super().__init__()

        # self.window_attention = WindowAttention(in_channel = 3, 
        #                            dim = 32, 
        #                            heads = 8, 
        #                            head_dim = 16, 
        #                            shifted = True, 
        #                            window_size = 4, 
        #                            relative_pos_embedding = True)

        self.attention_map = AttentionMap()
        self.reflectance_extractor =  UNet(input_feats=4, output_feats=3)
        self.illumination_extractor = UNet(input_feats=4, output_feats=3)
        self.illumination_enhancer = UNet(input_feats=3, output_feats=3)
        self.reflectance_enhancer = UNet(input_feats=3, output_feats=3)
    
    def forward(self, input_low_light, target_high_light, mode = 'train'):
        if mode == 'train':
            # pass through transformer
            input_low_light_attn = self.attention_map(input_low_light)
            target_high_light_attn = expected_attention_map(input_low_light, target_high_light)

            input_low_light_cat_attn = torch.cat([input_low_light, input_low_light_attn], dim=1)  # Shape: (B, 4, H, W)
            target_high_ligh_cat_attn = torch.cat([target_high_light, target_high_light_attn], dim=1)

            # extract reflectance
            reflectance_high_light = self.reflectance_extractor(input_low_light_cat_attn)
            reflectance_low_light = self.reflectance_extractor(target_high_ligh_cat_attn)

            # extract illumination
            illumination_low_light = self.illumination_extractor(input_low_light_cat_attn)
            illumination_high_light = self.illumination_extractor(target_high_ligh_cat_attn)

            # enhance illumination for low light image
            enhanced_illumination = self.illumination_enhancer(illumination_low_light)
            enhanced_reflectance = self.reflectance_enhancer(reflectance_low_light)

            return (
            input_low_light_attn,
            target_high_light_attn,
            reflectance_low_light,
            reflectance_high_light,
            illumination_low_light,
            illumination_high_light,
            enhanced_illumination,
            enhanced_reflectance,
        ) 

        if mode == 'eval':
            # pass through transformer
            input_low_light_attn = self.attention_map(input_low_light)
            input_low_light_cat_attn = torch.cat([input_low_light, input_low_light_attn], dim=1)  # Shape: (B, 4, H, W)

            # extract reflectance
            reflectance_low_light = self.reflectance_extractor(input_low_light_cat_attn)

            # extract illumination
            illumination_low_light = self.illumination_extractor(input_low_light_cat_attn)

            # enhance illumination for low light image
            enhanced_illumination = self.illumination_enhancer(illumination_low_light)
            enhanced_reflectance = self.reflectance_enhancer(reflectance_low_light)

            return enhanced_reflectance * enhanced_illumination