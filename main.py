import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from DatasetAndAugmentation.LowHighDataAugment import PairedTransforms
from DatasetAndAugmentation.LowHightDataset import LOLPairedDataset
import matplotlib.pyplot as plt
from model.model import Model, VAELoss
import torch.nn as nn
from utils import cal_brightness, batchly_show_pic, torch_type_adapt
from get_args import get_args
import yaml
import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import random
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
train_low_dir = "./LOL-v2/Synthetic/Train/Low"
# directory of high-light images
train_bright_dir = "./LOL-v2/Synthetic/Train/Normal"

# test
test_low_dir = "./LOL-v2/Synthetic/Test/Low"
test_bright_dir = "./LOL-v2/Synthetic/Test/Normal"


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
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)



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
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print("Loading model to device: {}".format(device))
        model = Model(model_args).to(device)
        model.load_state_dict(torch.load(train_savedir / "model.pth", map_location=device))

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
        
# loss = nn.L1Loss().to(device)
if model_args.get("use_vae", True):
    loss = VAELoss().to(device)
else:
    loss = torch.nn.L1Loss().to(device)
# learning rate


initial_lr = float(training_args['learning_rate'])
min_lr = float(training_args['learning_rate_min'])
gamma =float(training_args['gamma'])

def lr_lambda(epoch):
    return max(gamma ** epoch, min_lr / initial_lr)

optimizer = AdamW(model.parameters(), lr=initial_lr )
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
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

def train_epoch(model, train_loader, log, optimizer, loss, device, epoch_num):
    model.train()
    loss.train()
    qbar = tqdm(train_loader)
    log["epoch"] = epoch_num
    epoch_loss = 0
    epoch_iter_num = len(train_loader)
    show_iter = int(random.uniform(0, epoch_iter_num))
    for i, batch in enumerate(qbar):
        optimizer.zero_grad()
        input_low_light = torch_type_adapt(batch["low"], device).to(device)
        target_high_light = torch_type_adapt(batch["bright"], device).to(device)
        brightness_low = torch_type_adapt(batch["low_brightness_value"], device).to(device)
        brightness_high = torch_type_adapt(batch["bright_brightness_value"], device).to(device)    
        predict_high_light, mu, log_var = model(input_low_light, target_high_light ,input_brightness = brightness_low ,target_brightness=brightness_high ) 
        if isinstance(loss, VAELoss):
            l = loss(predict_high_light, target_high_light, mu, log_var)
        else:
            l = loss(predict_high_light, target_high_light)
        l.backward()

        # 🔐 Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        loss_value = l.to('cpu').detach().item()
        log["loss_iter"] = loss_value
        epoch_loss += loss_value
        qbar.set_postfix(log)
        # randomly show the image during the training
        if i == show_iter:
            try:
                batchly_show_pic(
                    input_low_light.to('cpu').detach(),
                    predict_high_light.to('cpu').detach(),
                    target_high_light.to('cpu').detach(),
                    brightness_high.to('cpu').detach(),
                    brightness_low.to('cpu').detach(),
                    train_transform, save_path= train_savedir/"training_results/train_epoch_{}_iter_{}.png".format(epoch_num, i))
            except:
                print("Error in saving the image")
    return model, log, optimizer, loss, device, epoch_num, epoch_loss, qbar

def test_epoch(model, test_loader, log, loss_fn, device, epoch_num, use_vae):
    model.eval()
    loss_fn.eval()
    total_loss = 0
    num_batches = 0

    for i, batch in enumerate(test_loader):
        input_low_light = torch_type_adapt(batch["low"], device).to(device)
        target_high_light = torch_type_adapt(batch["bright"], device).to(device)
        brightness_low = torch_type_adapt(batch["low_brightness_value"], device).to(device)
        brightness_high = torch_type_adapt(batch["bright_brightness_value"], device).to(device)    

        with torch.no_grad():
            pred, mu, log_var = model(
                input_low_light, target_high_light,
                input_brightness=brightness_low,
                target_brightness=brightness_high
            )

            if use_vae:
                l = loss_fn(pred, target_high_light, mu, log_var)
            else:
                l = loss_fn(pred, target_high_light)

            loss_value = l.to('cpu').detach().item()
            total_loss += loss_value
            num_batches += 1

            batchly_show_pic(
                input_low_light.to('cpu').detach(),
                pred.to('cpu').detach(),
                target_high_light.to('cpu').detach(),
                brightness_high.to('cpu').detach(),
                brightness_low.to('cpu').detach(),
                test_transform,
                save_path=train_savedir / f"training_results/test_epoch_{epoch_num}_iter_{i}.png"
            )

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return model, log, loss_fn, device, epoch_num, avg_loss
    

train_losses, test_losses = [], []
best_test_loss = float('inf')
best_model_path = train_savedir / "best_model.pth"
for epoch in range(training_args['epochs']):
    use_vae = model_args.get("use_vae", True)  # default to True if not specified

    # Train for one epoch
    model, log, optimizer, loss, device, epoch_num, epoch_loss, qbar = train_epoch(
        model, train_loader, log, optimizer, loss, device, epoch
    )

    # Evaluate on test set
    model, log, loss, device, epoch_num, test_loss = test_epoch(
        model, test_loader, log, loss, device, epoch, use_vae
    )

    # 📢 Print test loss
    print(f"[Epoch {epoch}] ✅ Test Loss: {test_loss:.4f}")

    # Saving the best model regarding to test loss
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"💾 Best model saved at epoch {epoch} with test loss {best_test_loss:.4f}")


    # Save model every 10 epochs
    # if epoch % 10 == 1:
    #     torch.save(model, train_savedir / "model.pth")
    #     print(f"💾 Model saved at: {train_savedir / 'model.pth'}")

    # Update learning rate
    scheduler.step()

    # Log and print training loss
    log['loss_epoch'] = epoch_loss / len(train_loader)
    qbar.set_postfix(log)
    print(f"[Epoch {epoch}] 🏋️ Train Loss: {log['loss_epoch']:.4f}")
    train_losses.append(log['loss_epoch'])
    test_losses.append(test_loss)

plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.title("Training Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


        


"""
****************************************

# Visualize the output of model

*********************************************
""" 
# batchly_show_pic(input_low_light, predict_high_light, target_high_light, brightness_high, brightness_low, train_transform)