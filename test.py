# imports
import os
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image as PILImage  # for image loading
from IPython.display import Image, display  # for displaying in notebook
from tqdm import tqdm
# Custom modules
from DatasetAndAugmentation.LowHighDataAugment import PairedTransforms
from DatasetAndAugmentation.LowHightDataset import LOLPairedDataset
from model.model import Model, VAELoss
from utils import batchly_show_pic, torch_type_adapt
from types import SimpleNamespace

from types import SimpleNamespace
from pathlib import Path
import torch
import yaml
from model.model import Model  # adjust based on your file structure

# Mount Google Drive


# ✅ Define root directory for your project in Drive
project_root = Path("./")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Step 1: Define baseline model args ===
retinexformer_args = SimpleNamespace(
    model_option=project_root / "train_results/retinexformer_baseline/train_option.yaml",  # adjust this path
    train_save_dir="retinexformer_baseline"
)

# === Step 2: Setup paths ===
train_savedir = project_root / "train_results" / retinexformer_args.train_save_dir
option_path = retinexformer_args.model_option
best_model_path = train_savedir / "model.pth"

# === Step 3: Load YAML and initialize model ===
yaml_info = yaml.safe_load(open(option_path, 'r'))
model_args = yaml_info["network"]

# Make sure this is set to False
model_args["use_vae"] = False  # ⬅️ Make sure the model runs in baseline mode

# === Step 4: Load model ===
retinexformer = Model(model_args).to(device)
retinexformer.load_state_dict(torch.load(best_model_path, map_location=device))
retinexformer.eval()

print("✅ RetinexFormer (no VAE) loaded and ready for evaluation!")


test_transform = PairedTransforms(image_size=(400, 600), train=False)
test_dataset = LOLPairedDataset(
    project_root / "LOL-v2/Synthetic/Test/Low",
    project_root / "LOL-v2/Synthetic/Test/Normal",
    transform=test_transform,
    train=False,
    brightness_calculation_option=model_args['brightness_calculation_option']
)


test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)
for i, batch in enumerate(test_loader):
    input_low = torch_type_adapt(batch["low"], device).to(device)
    target_high = torch_type_adapt(batch["bright"], device).to(device)
    brightness_low = torch_type_adapt(batch["low_brightness_value"], device).to(device)
    brightness_high = torch_type_adapt(batch["bright_brightness_value"], device).to(device)

    with torch.no_grad():
        pred, mu, log_var = retinexformer(
            input_low, target_high,
            input_brightness=brightness_low,
            target_brightness=brightness_high
        )

    save_path = train_savedir / f"test_result_batch_{i}.png"
    batchly_show_pic(
        input_low.to("cpu").detach(),
        pred.to("cpu").detach(),
        target_high.to("cpu").detach(),
        brightness_high.to("cpu").detach(),
        brightness_low.to("cpu").detach(),
        test_transform,
        save_path=save_path
    )

    # ✅ Compact display
    display(Image(filename=str(save_path), width=1200, height=1200))
    break

    
import torchvision
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np

def compute_psnr(image_test, image_true ):
    image_test = image_test.cpu().numpy().transpose(1, 2, 0)
    image_true  =  image_true .cpu().numpy().transpose(1, 2, 0)
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

all_retinex_psnr, all_retinex_ssim = [], []

retinexformer.eval()
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        input_low = torch_type_adapt(batch["low"], device).to(device)
        target_high = torch_type_adapt(batch["bright"], device).to(device)
        brightness_low = torch_type_adapt(batch["low_brightness_value"], device).to(device)
        brightness_high = torch_type_adapt(batch["bright_brightness_value"], device).to(device)

        pred, mu, log_var = retinexformer(
            input_low, target_high,
            input_brightness=brightness_low,
            target_brightness=brightness_high
        )

        psnr_batch, ssim_batch = compute_batch_metrics(pred, target_high)
        all_retinex_psnr.extend(psnr_batch)
        all_retinex_ssim.extend(ssim_batch)

# Report average scores
mean_retinex_psnr = np.mean(all_retinex_psnr)
mean_retinex_ssim = np.mean(all_retinex_ssim)

print(f"📊 Final Evaluation of Retinexformer on Test Set:")
print(f"🔸 PSNR: {mean_retinex_psnr:.2f} dB")
print(f"🔹 SSIM: {mean_retinex_ssim:.4f}")
