#1. Pick a larger model from torchvision.models to add to the list of experiments (for example, EffNetB3 or higher).
# How does it perform compared to our existing models?

import torch, torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torchinfo import summary
from going_modular import data_setup, engine

#setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#create a helper func to set seeds
def set_seeds(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# 1. Get data
import os
import zipfile

from pathlib import Path
import requests

def download_data(source,
                  destination,
                  remove_source: bool=True):
    #setup path to data folder
    data_path = Path('data/')
    image_path = data_path / destination

    if image_path.is_dir():
        print(f'{image_path} already exist')
    else:
        print(f'Did not find {image_path}, downloading')
        image_path.mkdir(parents=True, exist_ok=True)

        #download
        target_file = Path(source).name
        with open(data_path / target_file, 'wb') as f:
            request = requests.get(source)
            print(f'Downloading {target_file} from {source}')
            f.write(request.content)

        #unzip
        with zipfile.ZipFile(data_path / target_file, 'r') as zip_ref:
            print(f'Unzipping {target_file}')
            zip_ref.extractall(image_path)

        #remove .zip file
        os.remove(data_path / target_file)

    return image_path

image_path = download_data(source='https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip',
                           destination='pizza_steak_sushi')


import torchvision.models as models
effnetv2_s_weights = models.EfficientNet_V2_S_Weights.DEFAULT
effnetv2_s = models.efficientnet_v2_s(weights=effnetv2_s_weights)
# print(effnetv2_s)

from torchinfo import summary
# summary(model=effnetv2_s,
#         input_size=(1, 3, 224, 224))

from torch.utils.tensorboard import SummaryWriter

def create_writer(experiment_name,
                  model_name,
                  extra=None):
    from datetime import datetime
    import os
    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestap = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        #create log directory path
        log_dir = os.path.join('runs', timestap, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join('runs', timestap, experiment_name, model_name)

    print(f'[INFO] Created SummaryWriter, saving to: {log_dir}...')
    return SummaryWriter(log_dir=log_dir)


from tqdm.auto import tqdm
from going_modular.engine import train_step, test_step
writer = create_writer(experiment_name='test_experiment',
                       model_name='this_is_model_name',
                       extra='add_a_title')

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter): # new parameter to take in a writer
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        ### New: Use the writer parameter to track experiments ###
        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy",
                               tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc},
                               global_step=epoch)

            # Close the writer
            writer.close()
        else:
            pass
    ### End new ###

    # Return the filled results at the end of the epochs
    return results


#download 10 percent and 20 training
data_10_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                                     destination="pizza_steak_sushi")

data_20_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
                                     destination="pizza_steak_sushi_20_percent")

# setup training directory paths
train_dir_10_percent = data_10_percent_path / 'train'
train_dir_20_percent = data_20_percent_path / 'train'

# Setup testing directory paths (note: use the same test dataset for both to compare the results)
test_dir = data_10_percent_path / 'test'

from torchvision import transforms

# Create a transform to normalize data distribution to be inline with ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # values per colour channel [red, green, blue],
                                 std=[0.229, 0.224, 0.225]) # values per colour channel [red, green, blue]

#compose transforms into pipline
simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),#Turn the images into tensors with values between 0 & 1
    normalize
])

# Now let's create our DataLoaders using the create_dataloaders() function from data_setup.py we created in 05. PyTorch Going Modular section 2.
BATCH_SIZE = 32

#create 10% training and test dataloaders
train_dataloader_10_percent, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir_10_percent,
    test_dir=test_dir,
    transform=simple_transform,
    batch_size=BATCH_SIZE
)

#create 20% training and test data dataloaders
train_dataloader_20_percent, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir_20_percent,
    test_dir=test_dir,
    transform=simple_transform,
    batch_size=BATCH_SIZE
)

def create_model(model_name, out_features=len(class_names)):
    assert model_name == 'effnetb2' or model_name == 'effnetv2_s', 'model_name should be effnetb2 or effnetv2_s'
    if model_name=='effnetb2':
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        model = torchvision.models.efficientnet_b2(weights=weights)
        dropout = 0.3
        in_features = 1408
    else:
        weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        model = torchvision.models.efficientnet_v2_s(weights=weights)
        dropout = 0.2
        in_features = 1280

    #freeze the base model layer
    for param in model.features.parameters():
        param.requires_grad = False

    #update classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features, out_features=out_features)
    )

    #set seeds
    set_seeds()

    #set the model name
    model.name = model_name
    print(f'creating {model.name} feature extractor model')
    return model

effnetb2 = create_model(model_name='effnetb2')
# print(effnetb2)


#create epoch list
num_epochs = [5, 10]

#create model list
models = ['effnetb2', 'effnetv2_s']

#create dataloaders dictionary for various dataloaders
train_dataloaders = {'data_10_percent':train_dataloader_10_percent,
                     'data_20_percent':train_dataloader_20_percent}

from going_modular.utils import save_model
#set the random seed
set_seeds()

#keep track of expirement numbers
experiment_number = 0

# loop through each Dataloader
# for dataloader_name, train_dataloader in train_dataloaders.items():
#     #loop through each number of epochs
#     for epochs in num_epochs:
#         #loop through each name and create a new model based on the name
#         for model_name in models:
#             #info
#             experiment_number += 1
#             print(f'Experiment number {experiment_number}')
#             print(f'model {model_name}')
#             print(f'Dataloader {dataloader_name}')
#             print(f'number of epochs {epochs}')
#
#             #select the model
#             model = create_model(model_name=model_name).to(device) #instantiate a new model instance
#
#             #create a new loss and optimier for every model
#             loss_fn = nn.CrossEntropyLoss()
#             optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
#
#             #train target model with target dataloaders and track experiments
#             train(model=model,
#                   train_dataloader=train_dataloader,
#                   test_dataloader=test_dataloader,
#                   optimizer=optimizer,
#                   loss_fn=loss_fn,
#                   epochs=epochs,
#                   device=device,
#                   writer=create_writer(experiment_name=dataloader_name,
#                                        model_name=model_name,
#                                        extra=f'{epochs}_epochs'))
#
#             #save the model to file
#             save_filepath = f'07_{model_name}_{dataloader_name}_{epochs}_epochs.pth'
#             save_model(model=model,
#                        target_dir='models',
#                        model_name=save_filepath)
#             print('-'*50+'\n')


# 2. Introduce data augmentation to the list of experiments using the 20% pizza, steak, sushi training and test datasets, does this change anything?
#create data augmentation transform
from torchvision import transforms

#create transform to normalize data distribution to be inline with imagenet
data_aug_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    normalize
])

#create non data aug transform
no_data_aug_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

#updated create_dataloaders
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 0 #os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    train_transform: transforms.Compose,#new
    test_transform: transforms.Compose,#new
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=train_transform)
  test_data = datasets.ImageFolder(test_dir, transform=test_transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names

#create train and test directories
train_20_percent_dir = image_path / 'train'
test_20_percent_dir = image_path / 'test'

BATCH_SIZE = 32

#create train dataloader with augmentation
train_dataloader_20_percent_with_aug, test_dataloader_20_percent, class_names = create_dataloaders(train_dir=train_20_percent_dir,
                                                                                                              test_dir=test_20_percent_dir,
                                                                                                              train_transform=data_aug_transform,
                                                                                                              test_transform=no_data_aug_transform,
                                                                                                              batch_size=BATCH_SIZE
                                                                                                   )

#create train dataloader without data augmentation
train_dataloader_20_percent_without_aug, test_dataloader_20_percent, class_names = create_dataloaders(train_dir=train_20_percent_dir,
                                                                                                      test_dir=test_20_percent_dir,
                                                                                                      train_transform=no_data_aug_transform,
                                                                                                      test_transform=no_data_aug_transform,
                                                                                                      batch_size=BATCH_SIZE)

#visualize different samples from both dataloaders
def view_dataloader_images(dataloader, n=10):
    if n>10:
        print('having n higher than 10 will create messy plots, lowering to 10')
        n = 10
    imgs, labels = next(iter(dataloader))
    plt.figure(figsize=(16, 8))
    for i in range(n):
        #min max scale the image for display purposes
        targ_image = imgs[i]
        sample_min, sample_max = targ_image.min(), targ_image.max()
        sample_scaled = (targ_image - sample_min) / (sample_max - sample_min)

        #plot images with appropriate axes information
        plt.subplot(1, 10, i+1)
        plt.imshow(sample_scaled.permute(1, 2, 0))#resize matplotlib requirements
        plt.title(class_names[labels[i]])
        plt.axis(False)
    plt.show()

#check out samples with data augmentation
# view_dataloader_images(train_dataloader_20_percent_with_aug)

#check out samples without data augmentation
# view_dataloader_images(train_dataloader_20_percent_without_aug)

def create_effnetb2(out_features=len(class_names)):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)
    dropout = 0.3
    in_features=1408

    # Freeze the base model layers
    for param in model.features.parameters():
      param.requires_grad=False

    # Set the seeds
    set_seeds()

    # Update the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features,
                  out_features=out_features)
    ).to(device)

    # Set the model name
    model.name = "effnetb2"
    print(f"[INFO] Creating {model.name} feature extractor model...")
    return model

#create an EffnetB2 feature extractor
def create_effnetv2_s(out_features=len(class_names)):
    weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    model = torchvision.models.efficientnet_v2_s(weights=weights).to(device)
    dropout = 0.2
    in_features = 1280

    # Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Set the seeds
    set_seeds()

    # Update the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features,
                  out_features=out_features)
    ).to(device)

    # Set the model name
    model.name = "effnetv2_s"
    print(f"[INFO] Creating {model.name} feature extractor model...")
    return model

    #give the model name
    model.name = 'effnetb2'
    print(f'created new {model.name} model')
    return model


#run data aug vs no data aug experiments
#setup the number of epochs
epochs = [5, 10]

#create dataloaders dict for various dataloaders
train_dataloaders = {'data_20_percent_with_aug':train_dataloader_20_percent_with_aug,
                     'data_20_percent_without_aug':train_dataloader_20_percent_without_aug}

#create model
models = ['effnetv2_s']

from going_modular.utils import save_model
#set the random seed
set_seeds()

#keep track of expirement numbers
experiment_number = 0

#loop through each Dataloader
# for dataloader_name, train_dataloader in train_dataloaders.items():
#     #loop through each number of epochs
#     for epochs in num_epochs:
#         #loop through each name and create a new model based on the name
#         for model_name in models:
#             #info
#             experiment_number += 1
#             print(f'Experiment number {experiment_number}')
#             print(f'model {model_name}')
#             print(f'Dataloader {dataloader_name}')
#             print(f'number of epochs {epochs}')
#
#             #select the model
#             if model_name == "effnetb2":
#                 model = create_effnetb2()
#             else:
#                 model = create_effnetv2_s()
#
#             #create a new loss and optimier for every model
#             loss_fn = nn.CrossEntropyLoss()
#             optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
#
#             #train target model with target dataloaders and track experiments
#             train(model=model,
#                   train_dataloader=train_dataloader,
#                   test_dataloader=test_dataloader_20_percent, ##new
#                   optimizer=optimizer,
#                   loss_fn=loss_fn,
#                   epochs=epochs,
#                   device=device,
#                   writer=create_writer(experiment_name=dataloader_name,
#                                        model_name=model_name,
#                                        extra=f'{epochs}_epochs'))
#
#             #save the model to file
#             save_filepath = f'07_{model_name}_{dataloader_name}_{epochs}_epochs.pth'
#             save_model(model=model,
#                        target_dir='models',
#                        model_name=save_filepath)
#             print('-'*50+'\n')


# 3.Scale up the dataset to turn FoodVision Mini into FoodVision Big using the entire Food101 dataset from torchvision.models
#get Food101 dataset
import torchvision
from torchvision import transforms

#create transform to normalize data distribution to be inline with ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # values per colour channel [red, green, blue],
                                 std=[0.229, 0.224, 0.225]) # values per colour channel [red, green, blue]

#create transform pipline
simple_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
])

#download and transform Food101 data
train_data = torchvision.datasets.Food101(root='data',
                                       split='train',
                                       transform=simple_transform,
                                       download=True)

test_data = torchvision.datasets.Food101(root='data',
                                      split='test',
                                      transform=simple_transform,
                                      download=True)


#create dataloaders
import os
BATCH_SIZE = 256

train_dataloader_big = torch.utils.data.DataLoader(train_data,
                                                   shuffle=True,
                                                   batch_size=BATCH_SIZE,
                                                   num_workers=0,
                                                   pin_memory=True) #avoid copies of the data into and out of memory (for speed ups)

test_dataloader_big = torch.utils.data.DataLoader(test_data,
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE,
                                                  num_workers=0,
                                                  pin_memory=True)

#create model
effnetv2_s_weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
foodvision_big_model = torchvision.models.efficientnet_v2_s(weights=effnetv2_s_weights).to(device)

import torch.nn as nn
#freeze the base layers
for param in foodvision_big_model.features.parameters():
    param.requires_grad = False

#change the classifier head to suit 101 different classes
foodvision_big_model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features=1280, out_features=101) #101 output classes for Food101
).to(device)

# print(foodvision_big_model)

# summary(model=foodvision_big_model,
#         input_size=(1,3, 224,224))

foodvision_big_results = train(model=foodvision_big_model,
                               train_dataloader=train_dataloader_big,
                               test_dataloader=test_dataloader_big,
                               optimizer=torch.optim.Adam(params=foodvision_big_model.parameters(), lr=0.001),
                               loss_fn=nn.CrossEntropyLoss(),
                               epochs=5,
                               device=device,
                               writer=create_writer(experiment_name='food101_all_data',
                                                    model_name='foodvision_big',
                                                    extra=f'{epochs} epochs'
                                ))






