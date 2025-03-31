import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from DatasetAndAugmentation.LowHighDataAugment import PairedTransforms
from DatasetAndAugmentation.LowHightDataset import LOLPairedDataset
import matplotlib.pyplot as plt
from model.model import Model
import torch.nn as nn
from utils import cal_brightness, batchly_show_pic
from get_args import get_args



"""
****************************************

# Define the dir of the dataset

*********************************************
""" 


# direction of the dataset
dataset_dir = "/path/to/dataset"
# directory of low-light images
train_low_dir = "./LOLdataset/train/low"
# directory of high-light images
train_bright_dir = "./LOLdataset/train/high"

# test
test_low_dir = "./LOLdataset/test/low"
test_bright_dir = "./LOLdataset/test/high"


"""
****************************************

# create the train/test pic transformer, dataset and dataLoader

*********************************************
""" 

# create transform class to transform the image into tensor
train_batch_size = 5
train_transform = PairedTransforms(image_size=(400, 600), train=True)
train_dataset = LOLPairedDataset(train_low_dir, train_bright_dir, transform=train_transform, train=True)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)


test_transform = PairedTransforms(image_size=(400, 600), train=False)
test_dataset = LOLPairedDataset(test_low_dir, test_bright_dir, transform=test_transform, train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



"""
****************************************

# Initialize the model

*********************************************
""" 

model = Model() # Currently directly output the input
loss = nn.MSELoss()


"""
****************************************

# Pipline of training Process

*********************************************
""" 

for batch in train_loader:
    input_low_light = batch["low"]
    target_high_light = batch["bright"]
    brightness_low = batch["low_brightness_value"]
    brightness_high = batch["bright_brightness_value"]
    predict_high_light = model(input_low_light, degree_of_brightness=-1)
    l = loss(predict_high_light, target_high_light)
    break

"""
****************************************

# Visualize the output of model

*********************************************
""" 
batchly_show_pic(input_low_light, predict_high_light, target_high_light, brightness_high, brightness_low, train_transform)