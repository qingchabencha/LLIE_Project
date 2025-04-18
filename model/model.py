import torch.nn as nn
from model.retinex_extractor import RetinexExtractor
from model.window_attention import WindowAttention

class RELLIE(nn.Module):
    def __init__(self):
        super().__init__()

        self.window_attention = WindowAttention(in_channel = 3, 
                                   dim = 32, 
                                   heads = 8, 
                                   head_dim = 16, 
                                   shifted = True, 
                                   window_size = 4, 
                                   relative_pos_embedding = True)
        
        self.reflectance_extractor =  RetinexExtractor()
        self.illumination_extractor = RetinexExtractor()
        self.illumination_enhancer = RetinexExtractor()
        self.reflectance_enhancer = RetinexExtractor()
    
    def forward(self, input_low_light, target_high_light, mode):
        if mode == 'train':
            # pass through transformer
            transformed_low_light_img = self.window_attention(input_low_light)
            transformed_high_light_img = self.window_attention(target_high_light)

            # extract reflectance
            reflectance_high_light = self.reflectance_extractor(transformed_high_light_img)
            reflectance_low_light = self.reflectance_extractor(transformed_low_light_img)

            # extract illumination
            illumination_low_light = self.illumination_extractor(transformed_low_light_img)
            illumination_high_light = self.illumination_extractor(transformed_high_light_img)

            # enhance illumination for low light image
            enhanced_illumination = self.illumination_enhancer(illumination_low_light)
            enhanced_reflectance = self.reflectance_enhancer(reflectance_low_light)

            return reflectance_low_light, reflectance_high_light, illumination_low_light, illumination_high_light, enhanced_illumination, enhanced_reflectance

        if mode == 'eval':
            # pass through transformer
            transformed_low_light_img = self.window_attention(input_low_light)

            # extract reflectance
            reflectance_low_light = self.reflectance_extractor(transformed_low_light_img)

            # extract illumination
            illumination_low_light = self.illumination_extractor(transformed_low_light_img)

            # enhance illumination for low light image
            enhanced_illumination = self.illumination_enhancer(illumination_low_light)
            enhanced_reflectance = self.reflectance_enhancer(reflectance_low_light)

            return enhanced_reflectance * enhanced_illumination