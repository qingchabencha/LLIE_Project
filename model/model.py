import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
import model.module_retindexformer as module_retindexformer
from torchvision.models import vgg16
from torchvision.transforms.functional import rgb_to_grayscale


class Model(torch.nn.Module):
    """
    A class for the model.
    """

    def __init__(self, model_args):
        """
        Initialize the model with the given model and device.

        Args:
            model_args (torch.nn.Module): Argument for initlialize the model, use ./Options/xxx.yml to see the details.
        """
        super(Model, self).__init__()
        self.brightness_calculation_option = model_args['brightness_calculation_option']

        # ************************************************************************************************

        # The first part, Illumination Estimation part

        # Two option for Light-up Map and Light-up Feature creation
        # 1. Original method in paper: con2d(4, hidden, kernel_size=1, stride=1, padding=0)
        # 2. New method: Fast Fourier Convolution, increase the receptive field of the convolutional layer

        # *************************************************************************************************

        self.illuminattion_estimator = None
        if model_args['model_illumination_estimator_model'] == "CNN":
            self.illuminattion_estimator = module_retindexformer.Illumination_Estimator(
                model_type=model_args["model_illumination_estimator_model"],
                hidden_channel_dim=model_args["IE_feature_channels"],
                CNN_Kernel_size=model_args["CNN_Kernel_size"],
                CNN_Feature_active=model_args["CNN_Feature_active"],
            )

        # TODO: implement the FFC method
        elif model_args['model_illumination_estimator_model'] == "FFC":
            pass
        else:
            raise ValueError("model_illumination_estimator_model should be CNN or FFC the method {} is not supported".format(
                model_args['model_illumination_estimator_model']))
        # ***********************************************************************************************

        # Second part, value generation part

        # ***********************************************************************************************
        
        self.use_vae = model_args.get('use_vae', True)

        if self.use_vae:
            self.colorRestorator = module_retindexformer.ColorComplementorVAE(
                lattent_dim=model_args["latent_dim"],
                encode_layer=model_args["latent_encode_layer"],
                feature_in = model_args["IE_feature_channels"] + 3
            )

        
        
        # ***********************************************************************************************

        # Third part, Denoise part

        # ***********************************************************************************************

        self.denoise = module_retindexformer.Denoiser(
            in_dim=3,
            out_dim=3,
            dim=model_args["IE_feature_channels"],
            num_blocks=model_args['number_blocks'],
            level=len(model_args['number_blocks'])-1
        )

    def mixup(self, low, high):
        # TODO: implement the mixup function
        """
        Randomly mix two images with different level of alpha rate? (不同透明度合成一张图片)

        Args:
            low (_type_): low light image torch (B, C, H, W)
            high (_type_): high light image torch (B, C, H, W)

        Returns:
            _type_: Low, High
        """
        return low, high
    
    

    def forward(self, low, target_pic ,input_brightness=-1, target_brightness=-1):
        """
        Forward pass of the model.

        Args:
            low (torch.Tensor): The original low-light image.
            degree_of_brightness (int, optional): The degree of brightness adjustment. Defaults to -1.

        """
        # calculate the illumination of each pixel in the image
        lp = utils.cal_brightness_torch(
            low, average=False, option=self.brightness_calculation_option)
        lp = lp.unsqueeze(1)

        # Do Illumination Estimation
        illu_fea, illu_map = self.illuminattion_estimator(low, lp)
        # 添加信息后在提亮
        Lit_up_img = illu_map * low + low

        # pixel has insufficient illumination might directly set to 0,0,0, with no color to light up
        # TODO::implement VAE here, according to illumination prior & input image to create color for underexposed area
        if self.use_vae:
            restored_img, mu, log_var = self.colorRestorator(Lit_up_img, illu_fea, target_pic)
        else:
            restored_img = Lit_up_img
            mu, log_var = None, None

        
        # Denoise the artifcat & noise (e.g., ISO noise) in the image
        output_img = self.denoise(restored_img, illu_fea)

        return output_img, mu, log_var


class VAELoss(torch.nn.Module):
    """
    A class for the VAE loss.
    """
    def __init__(self):
        """
        Initialize the VAE loss with the given model and device.

        Args:
            model_args (torch.nn.Module): Argument for initlialize the model, use ./Options/xxx.yml to see the details.
        """
        super(VAELoss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()
    def forward(self, output, target, mu, log_var):
        """
        Forward pass of the VAE loss.

        Args:
            output (torch.Tensor): The original low-light image.
            target (torch.Tensor): The target high-light image.
            mu (torch.Tensor): The mean of the latent variable.
            log_var (torch.Tensor): The log variance of the latent variable.

        """
        
        # Reconstruction loss
        recon_loss = self.l1_loss(output, target)
        # KL divergence loss
        if mu is not None:
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        else:
            kl_loss = 0.0
        return recon_loss + kl_loss
