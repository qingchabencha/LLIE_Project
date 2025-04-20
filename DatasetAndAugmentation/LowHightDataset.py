import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import cal_brightness
from DatasetAndAugmentation.LowHighDataAugment import PairedTransforms

class LOLPairedDataset(Dataset):
    """
    A dataset class for loading paired low-light and bright images.

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, low_dir, bright_dir, transform:PairedTransforms=None, train=True, brightness_calculation_option=1):
        """
        Initialize the dataset with directories for low-light and bright images.

        Args:
            low_dir (str): the directory containing low-light images.
            bright_dir (_type_): the directory containing bright images.
            transform (_type_, optional): data augmentation and transformation class. Defaults to None.
        """
        self.low_paths = sorted(list(Path(low_dir).glob("*.png")))
        self.bright_paths = sorted(list(Path(bright_dir).glob("*.png")))
        assert len(self.low_paths) == len(self.bright_paths), "Mismatch in image pairs"
        self.transform = transform
        self.train = train
        self.brightness_calculation_option = brightness_calculation_option
        if transform:
            if self.train != transform.train:
                raise ValueError("The train flag in the dataset and transform should match.")
        else:
            Warning("No transform provided. Data augmentation and create tensor will not be applied.")

    def __getitem__(self, idx):
        low = Image.open(self.low_paths[idx]).convert("RGB")
        bright = Image.open(self.bright_paths[idx]).convert("RGB")
        
        # include augmentation and transform PIL class to tensor
        if self.transform:
            low, bright = self.transform(low, bright)
        return {"low": low, 
                "bright": bright,
                "low_brightness_value": cal_brightness(self.transform.tensor_to_PIL(low), average=True, option=self.brightness_calculation_option), 
                "bright_brightness_value": cal_brightness(self.transform.tensor_to_PIL(bright ), average=True, option=self.brightness_calculation_option)}

    def __len__(self):
        return len(self.low_paths)
