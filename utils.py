import PIL.PngImagePlugin
import numpy as np
from PIL import Image
import PIL
import matplotlib.pyplot as plt
from DatasetAndAugmentation.LowHighDataAugment import PairedTransforms
import yaml
import torch
import torchvision
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from tqdm import tqdm
from model.model import Model
import collections

def compute_psnr(image_test, image_true ):
    image_test = image_test.cpu().numpy().transpose(1, 2, 0)
    image_true = image_true.cpu().numpy().transpose(1, 2, 0)
    return compare_psnr(image_test=image_test, image_true =image_true , data_range=1.0)

def compute_ssim(img1, img2):
    img1 = img1.cpu().numpy().transpose(1, 2, 0)
    img2 = img2.cpu().numpy().transpose(1, 2, 0)
    return compare_ssim(img1, img2, multichannel=True, data_range=1.0, win_size=3)


def compute_batch_metrics(pred_batch, target_batch):
    psnr_list = []
    ssim_list = []
    for pred_img, target_img in zip(pred_batch, target_batch):
        psnr = compute_psnr(image_test = pred_img, image_true = target_img)
        ssim = compute_ssim(pred_img, target_img)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
    return psnr_list, ssim_list


def cal_brightness(image: PIL.PngImagePlugin.PngImageFile, average=True, option=1):
    """
    Calculate the brightness of each pixel in the image.
    :param image: PIL Image
    :return: numpy array of brightness values
    """
    # REFERENCE:
    # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
    # https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
    
    ##########################################################################################
    
    # Formula to calculate brightness:
    # L = 0.299 * R + 0.587 * G + 0.114 * B
    # L = sqrt( 0.299*R^2 + 0.587*G^2 + 0.114*B^2 )
    
    ###########################################################################################

    # the brightness can be calculated by direct tranform from RGB to L (grey scale) to evalute.
    
    
    # convert to one single channel
    gray_image = image.convert("L") # # 0.299 * R + 0.587 * G + 0.114 * B
    
    # Convert the grayscale image to a numpy array
    gray_array = np.array(gray_image)
    
    if option == 2: # ! maybe wrong
        # Calculate brightness using the second formula
        gray_array = np.sqrt(gray_array ** 2 + gray_array ** 2 + gray_array ** 2)
    
    if average:
        return np.mean(gray_array)
    else:
        return gray_array

def cal_brightness_torch(input: torch.Tensor, average=False, option=1):
    if option == 1:
        return input[:,0,...] * 0.299 + input[:,0,...] * 0.587 + input[:,0,...] * 0.114
    elif option == 2: # ! maybe wrong
        return torch.sqrt(input[..., 0] ** 2 * 0.299 + input[..., 1] ** 2 * 0.587 + input[..., 2] ** 2 * 0.114)
    
    
def batchly_show_pic(input_batch, 
                     predict_batch,
                     target_batch,
                     illumination_level_high,
                     illumination_level_low,
                     aug_transform:PairedTransforms,
                     save_path=None,
                     ):
    """
    Show the input, predicted and target images in a batch.

    Args:
        input_batch (tensor): _input batch of low-light images_
        predict_batch (_type_): the predicted batch of high-light images
        target_batch (_type_): the target batch of high-light images
        illumination_level_high (_type_): the brightness level of the high-light images
        illumination_level_low (_type_): the brightness level of the low-light images
        aug_transform (PairedTransforms): transform class to convert the tensor to PIL image
    """
    batch_size = input_batch.shape[0]
    fig, axes = plt.subplots(3, batch_size, figsize=(3 * batch_size, 9))
    if batch_size == 1:
        axes = np.expand_dims(axes, axis=0).T
    for i in range(batch_size):
        predict_high_light_PIL, target_high_light_PIL = aug_transform.tensor2PIL(predict_batch[i,...],  target_batch[i,...])
        input_low_light_PIL, _  = aug_transform.tensor2PIL(input_batch[i,...], None)
        
        brightness_level_low = illumination_level_high[i]
        brightness_level_high =  illumination_level_low[i]
        # 第1行: Picture input into the model
        axes[0, i].imshow(input_low_light_PIL)
        axes[0, i].set_title(f"Input {i+1}\nBrightness: {brightness_level_low:.2f}")
        axes[0, i].axis('off')

        # 第2行: Picture output from the model, augmented by the model
        axes[1, i].imshow(predict_high_light_PIL)
        axes[1, i].set_title(f"Predicted {i+1}\nBrightness: {brightness_level_high:.2f}")
        axes[1, i].axis('off')

        # 第3行：The target picture
        axes[2, i].imshow(target_high_light_PIL)
        axes[2, i].set_title(f"Target {i+1}\nBrightness: {brightness_level_high:.2f}")
        axes[2, i].axis('off')
    if save_path is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    test_high_pic = Image.open("./LOLdataset/test/high/1.png")
    test_low_pic = Image.open("./LOLdataset/test/low/1.png")
    cal_brightness(test_high_pic)
    print(f"light of the high image: {cal_brightness(test_high_pic)}")
    print(f"light of the low image: {cal_brightness(test_low_pic)}")
    
    
def torch_type_adapt(input:torch.Tensor, device:torch.device) -> torch.Tensor:
    """
    Adapt the input tensor to the device type. if the device is 'mps', convert the input tensor to float32.
    :param input: Input tensor
    :param device: Device type (e.g., 'cpu', 'cuda', 'mps')
    
    :return: Adapted input tensor
    """
    if device.type == 'mps':
        input = input.float()
    return input


def valid_load_model(project_root, args, device):
    train_savedir = project_root / "train_results" / args.train_save_dir
    model_path = train_savedir / "model.pth"
    option_path = args.model_option
    yaml_info = yaml.safe_load(open(option_path, 'r'))
    model_args = yaml_info["network"]
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load the model
    model = torch.load(model_path, map_location=device)
    if isinstance(model, collections.OrderedDict):
        model = Model(model_args).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, model_args, device, train_savedir


def valid_data(model, dataset_loader, brightness_threshold=torch.inf, device=None):
    """
    Validate the model on the dataset and save the results.
    :param model: The model to be validated
    :param dataset: The dataset to be validated on
    :param brightness_threshold: The brightness threshold for filtering images
    :param device: The device to run the model on (e.g., 'cpu', 'cuda', 'mps')
    """
    
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # Set the model to evaluation mode
    model.eval()
    all_vae_psnr, all_vae_ssim = [], []
    qualified_sample_count = 0 
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataset_loader)):
            input_low = torch_type_adapt(batch["low"], device).to(device)
            target_high = torch_type_adapt(batch["bright"], device).to(device)
            brightness_low = torch_type_adapt(batch["low_brightness_value"], device).to(device)
            brightness_high = torch_type_adapt(batch["bright_brightness_value"], device).to(device)

            mask = (brightness_low.view(-1) < brightness_threshold)
            if mask.sum() == 0:
                continue  # skip this batch if no qualifying samples
            
            # Count unmasked samples
            qualified_sample_count += mask.sum().item()
            
            # Apply the mask
            input_low = input_low[mask]
            target_high = target_high[mask]
            brightness_low = brightness_low[mask]
            brightness_high = brightness_high[mask]
            
            # Forward pass
            pred, mu, log_var = model(
                input_low, target_high,
                input_brightness=brightness_low,
                target_brightness=brightness_high
            )
            # Compute PSNR & SSIM
            psnr_batch, ssim_batch = compute_batch_metrics(pred, target_high)
            all_vae_psnr.extend(psnr_batch)
            all_vae_ssim.extend(ssim_batch)
        
    return all_vae_psnr, all_vae_ssim, qualified_sample_count  
    