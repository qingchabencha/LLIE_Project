import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LOLPairedDataset(Dataset):
    """
    A dataset class for loading paired low-light and bright images.

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, low_dir, bright_dir, transform=None, train=True):
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

    def __getitem__(self, idx):
        low = Image.open(self.low_paths[idx]).convert("RGB")
        bright = Image.open(self.bright_paths[idx]).convert("RGB")
        
        if self.transform:
            low, bright = self.transform(low, bright)
        return {"low": low, "bright": bright}

    def __len__(self):
        return len(self.low_paths)
