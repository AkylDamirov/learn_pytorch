# 0. Getting setup
import torch, torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torchinfo import summary

# Try to import the going_modular directory, download it from GitHub if it doesn't work
try:
    from going_modular import data_setup, engine
except ImportError:
    import os
    import subprocess
    print('Could not find going modular scripts.. downloading them from github')
    subprocess.run(['git', 'clone', 'https://github.com/mrdbourke/pytorch-deep-learning'], check=True)
    os.rename('pytorch-deep_learning/going_modular', 'going_modular')
    subprocess.run(['rm', '-rf', 'pytorch-deep-learning'], check=True)

    from going_modular import data_setup, engine

#setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Get data
# Before we can start to use transfer learning, we'll need a dataset.
# To see how transfer learning compares to our previous attempts at model building, we'll download the same dataset we've been using for FoodVision Mini.
# Let's write some code to download the pizza_steak_sushi.zip dataset from the course GitHub and then unzip it.

import os
import zipfile
from pathlib import Path
import requests

#setup path to data folder

data_path = Path('data/')
image_path = data_path / 'pizza_steak_sushi'
# If the image folder doesn't exist, download it and prepare it...

# if image_path.is_dir():
#     print(f'{image_path} is exist')
# else:
#     print(f'Did not find {image_path} directory, creating one...')
#     image_path.mkdir(parents=True, exist_ok=True)
#
#     #download
#     with open(data_path / 'pizza_steak_sushi.zip', 'wb') as f:
#         request = requests.get('https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip')
#         print('Downloading...')
#         f.write(request.content)
#
#     #unzip
#     with zipfile.ZipFile(data_path / 'pizza_steak_sushi.zip', 'r') as zip_ref:
#         print('Unzipping')
#         zip_ref.extractall(image_path)
#
#     #remove zipfile
#     os.remove(data_path / 'pizza_steak_sushi.zip')


#setup dirs
train_dir = image_path / 'train'
test_dir = image_path / 'test'

# 2. Create Datasets and DataLoaders
# When using a pretrained model, it's important that your custom data going into the model is prepared in the same way as the original training data that went into the model.
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


manual_transforms = transforms.Compose([
    transforms.Resize((224,224,)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create training and testing DataLoaders as well as get a list of class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=manual_transforms,
                                                                               batch_size=32)

# Get a set of pretrained model weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet

# Get the transforms used to create our pretrained weights
auto_transforms = weights.transforms()

# We can use auto_transforms to create DataLoaders with create_dataloaders() just as before.
# Create training and testing DataLoaders as well as get a list of class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=auto_transforms,
                                                                               batch_size=32)

# 3.2 Setting up a pretrained model
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# 3.3 Getting a summary of our model with torchinfo.summary()
# summary(model=model,
#         input_size=(32, 3, 224, 224),
#         col_names=['input_size', 'output_size', 'num_params', 'trainable'],
#         col_width=20,
#         row_settings=['var_names'])


# 3.4 Freezing the base model and changing the output layer to suit our needs
# For parameters with requires_grad=False, PyTorch doesn't track gradient updates and in turn, these parameters won't be changed by our optimizer during training.
# In essence, a parameter with requires_grad=False is "untrainable" or "frozen" in place.

# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.features.parameters():
    param.requires_grad=False

# And we'll keep in_features=1280 for our Linear output layer but we'll change the out_features value to the length
# of our class_names (len(['pizza', 'steak', 'sushi']) = 3).
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=output_shape,
                    bias=True)).to(device)

# # Do a summary *after* freezing the features and changing the output classifier layer (uncomment for actual output)
# print(summary(model,
#         input_size=(32, 3, 224, 224),
#         verbose=0,
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# ))

# 4. Train model
#define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Note: We're only going to be training the parameters classifier here as all of the other parameters in our model have been frozen.
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#start the timer
# from timeit import default_timer as timer
# start_time = timer()
#
# #setup training and save the results
# if __name__ == '__main__':
#     results = engine.train(model=model,
#                            train_dataloader=train_dataloader,
#                            test_dataloader=test_dataloader,
#                            optimizer=optimizer,
#                            loss_fn=loss_fn,
#                            epochs=5,
#                            device=device)
#
#     end_time = timer()
#     print(f'Total training time {end_time-start_time:.3f} seconds')
