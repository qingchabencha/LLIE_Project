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
import yaml
import torch
from tqdm import tqdm
from torch.optim import AdamW
args = get_args()


train_store_parent = Path("./train_results")
if not train_store_parent.exists():
    train_store_parent.mkdir(parents=True, exist_ok=True)
train_savedir = Path(args.train_save_dir)
train_savedir = train_store_parent / train_savedir 
if not train_savedir.exists():
    os.makedirs(train_savedir)
    create_model = True
else:
    create_model = False
# store the images during the training
if not (train_savedir/"training_results").exists():
    os.makedirs(train_savedir/"training_results")
    



# read yaml file and get the model args
yaml_info = yaml.safe_load(open(args.model_option, 'r'))
model_args = yaml_info['network']
training_args = yaml_info['training']

device='cpu'
if model_args['device'] == "cpu":
    device = torch.device("cpu")
elif model_args['device'] == 'mps':
    device = torch.device("mps")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
train_batch_size = training_args['batch_size']
train_transform = PairedTransforms(image_size=(400, 600), train=True)
train_dataset = LOLPairedDataset(train_low_dir,
                                 train_bright_dir,
                                 transform=train_transform, 
                                 train=True, 
                                 brightness_calculation_option=model_args['brightness_calculation_option'])
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)


test_transform = PairedTransforms(image_size=(400, 600), train=False)
test_dataset = LOLPairedDataset(test_low_dir, 
                                test_bright_dir, 
                                transform=test_transform, 
                                train=False, 
                                brightness_calculation_option=model_args['brightness_calculation_option'])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



"""
****************************************

# Initialize the model

*********************************************
""" 
if create_model:
    print("Create a new model")
    model = Model(model_args).to(device) # Currently directly output the input
    yaml.safe_dump(yaml_info, open(train_savedir / "train_option.yaml", 'w'), default_flow_style=False)
else:
    try:
        print("Load the model from the directory {}".format(train_savedir))
        model = torch.load(train_savedir / "model.pth")
        yaml_info = yaml.safe_load(open(train_savedir / "train_option.yaml", 'r'))
        model_args = yaml_info['network']
        training_args = yaml_info['training']
        # yaml_info = yaml.safe_load(open(args.model_option, 'r'))
        # model_args = yaml_info['network']
        # training_args = yaml_info['training']
    except:
        print("No model found in the directory, create from scratch")
        model = Model(model_args).to(device) # Currently directly output the input
        yaml.safe_dump(yaml_info, open(train_savedir / "train_option.yaml", 'w'), default_flow_style=False)
        
loss = nn.L1Loss().to(device)
optimizer = AdamW(model.parameters(), lr=float(training_args['learning_rate']))

"""
****************************************

# Pipline of training Process

*********************************************
""" 

log = {
    "loss_iter": 0,
    "epoch": 0,
    "loss_epoch": 0,
}

valid_ = training_args['valid_per_iter']

for epoch in range(training_args['epochs']):
    qbar = tqdm(train_loader)
    log["epoch"] = epoch
    epoch_loss = 0
    for i, batch in enumerate(qbar):
        optimizer.zero_grad()
        input_low_light = batch["low"].to(device)
        target_high_light = batch["bright"].to(device)
        brightness_low = batch["low_brightness_value"].to(device)
        brightness_high = batch["bright_brightness_value"].to(device)
        predict_high_light = model(input_low_light, input_brightness = brightness_low ,target_brightness=brightness_high )
        l = loss(predict_high_light, target_high_light)
        l.backward()
        optimizer.step()
        loss_value = l.to('cpu').detach().item()
        log["loss_iter"] = loss_value
        epoch_loss += loss_value
        qbar.set_postfix(log)
        
        # check the result of the training # TODO: Write additional function for saving the model
        valid_ -= 1
        if valid_ <= 0:
            batchly_show_pic(
                input_low_light.to('cpu').detach(),
                predict_high_light.to('cpu').detach(),
                target_high_light.to('cpu').detach(),
                brightness_high.to('cpu').detach(),
                brightness_low.to('cpu').detach(),
                train_transform, save_path= train_savedir/"training_results/epoch_{}_iter_{}.png".format(epoch, i))
            valid_ = training_args['valid_per_iter']
    if epoch % 10 == 1:
        torch.save(model, train_savedir / "model.pth")
        print("model saved in {}".format(train_savedir / "model.pth"))
    log['loss_epoch'] = epoch_loss / len(train_loader)
    qbar.set_postfix(log)
    print("epoch: {}, loss: {}".format(epoch, log['loss_epoch']))
        


"""
****************************************

# Visualize the output of model

*********************************************
""" 
batchly_show_pic(input_low_light, predict_high_light, target_high_light, brightness_high, brightness_low, train_transform)