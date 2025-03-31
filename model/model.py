import torch.nn
import torch
import utils
import model.module_retindexformer as module_retindexformer


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

        # Second part, Denoise part

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
    
    

    def forward(self, low, input_brightness=-1, target_brightness=-1):
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
        # pixel has insufficient illumination might directly set to 0,0,0, with no color to light up
        Lit_up_img = illu_map * low + low

        # Denoise the artifcat & noise (e.g., ISO noise) in the image

        output_img = self.denoise(Lit_up_img, illu_fea)

        return output_img
