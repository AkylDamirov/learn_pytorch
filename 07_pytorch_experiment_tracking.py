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

# print(image_path)
# 2. Create Datasets and DataLoaders
# Setup directories
train_dir = image_path / 'train'
test_dir = image_path / 'test'

# Setup ImageNet normalization levels (turns all images into similar distribution as ImageNet)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Create transform pipeline manually
manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])
# print(f"Manually created transforms: {manual_transforms}")

#create dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms,
    batch_size=32
)

# 2.2 Create DataLoaders using automatically created transforms
# Let's now see what the same transformation pipeline looks like but this time by using automatic transforms.

#setup dirs
train_dir = image_path / 'train'
test_dir = image_path / 'test'

# Setup pretrained weights (plenty of these available in torchvision.models)
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

# Get transforms from weights (these are the transforms that were used to obtain the weights)
automatic_transforms = weights.transforms()

#create dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=automatic_transforms,
    batch_size=32
)

# 3. Getting a pretrained model, freezing the base layers and changing the classifier head
# Setup the model with the pretrained weights and send it to the target device
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# Freeze all base layers by setting requires_grad attribute to False
for param in model.features.parameters():
    param.requires_grad = False

# Since we're creating a new layer with random weights (torch.nn.Linear),
# let's set the seeds
set_seeds()

#update the classifier head to suit our problem
model.classifier = torch.nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280,
              out_features=len(class_names),
              bias=True).to(device)
)

# Base layers frozen, classifier head changed, let's get a summary of our model with torchinfo.summary().
from torchinfo import summary

# print(summary(model,
#         input_size=(32, 3, 224, 224,),
#         verbose=0,
#         col_names=['input_size', 'output_size', 'num_params', 'trainable'],
#         col_width=20,
#         row_settings=['var_names']))

# 4. Train model and track results
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Adjust train() function to track results with SummaryWriter()
from torch.utils.tensorboard import SummaryWriter

#create a writer with all default settings
writer = SummaryWriter()

from typing import Dict, List
from tqdm.auto import tqdm

from going_modular.engine import train_step, test_step

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs,
          device: torch.device):
    #create empty results dictionary
    results = {'train_loss':[],
               'train_acc':[],
               'test_loss':[],
               'test_acc':[]}

    #loop through training and testing steps for number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)

        test_loss, test_acc = test_step(model,
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

        #update results dictionary
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

        ### New: Experiment tracking ###
        # Add loss results to SummaryWriter
        writer.add_scalars(main_tag='Loss',
                           tag_scalar_dict={'train_loss':train_loss,
                                            'test_loss':test_loss},
                           global_step=epoch)

        #add accuracy results to SummaryWriter
        writer.add_scalars(main_tag='Accuracy',
                           tag_scalar_dict={'train_acc':train_acc,
                                            'test_acc':test_acc},
                           global_step=epoch)

        # Track the PyTorch model architecture
        writer.add_graph(model=model,
                         input_to_model=torch.rand(32, 3, 224, 224).to(device))

    #close the writer
    writer.close()

    return results

# Train model
# Note: Not using engine.train() since the original script isn't updated to use writer
# set_seeds()
# results = train(model=model,
#                 train_dataloader=train_dataloader,
#                 test_dataloader=test_dataloader,
#                 optimizer=optimizer,
#                 loss_fn=loss_fn,
#                 epochs=5,
#                 device=device)

# print(results)
# 5. View our model's results in TensorBoard
# tensorboard --logdir=runs

# 6. Create a helper function to build SummaryWriter() instances
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

#create an example writer
# example_writer = create_writer(experiment_name='data_10_percent',
#                                model_name='effnetb0',
#                                extra='5_epochs')


# 6.1 Update the train() function to include a writer parameter

from typing import Dict, List
from tqdm.auto import tqdm

#add writer parameter to train()
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

# 7. Setting up a series of modelling experiments
# 7.3 Download different datasets
#download 10 percent and 20 training
data_10_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                                     destination="pizza_steak_sushi")

data_20_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
                                     destination="pizza_steak_sushi_20_percent")

# We'll create different training directory paths but we'll only need one testing directory path since all experiments will be using the same test dataset (the test dataset from pizza, steak, sushi 10%).

# setup training directory paths
train_dir_10_percent = data_10_percent_path / 'train'
train_dir_20_percent = data_20_percent_path / 'train'

# Setup testing directory paths (note: use the same test dataset for both to compare the results)
test_dir = data_10_percent_path / 'test'

#check
# print(f'train 10 percent {train_dir_10_percent}')
# print(f'train 20 percent {train_dir_20_percent}')
# print(f'testing directory {test_dir}')

# 7.4 Transform Datasets and create DataLoaders
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

# 7.5 Create feature extractor models
import torchvision
from torchinfo import summary

#create an instance of Effnetb2 with pretrained weights
effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
effnetb2 = torchvision.models.efficientnet_b2(weights=effnetb2_weights)

#  Get a summary of standard EffNetB2 from torchvision.models (uncomment for full output)
# summary(model=effnetb2,
#         input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
#         # col_names=["input_size"], # uncomment for smaller output
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# )
#
# Get the number of in_features of the EfficientNetB2 classifier layer
# print(f"Number of in_features to final layer of EfficientNetB2: {len(effnetb2.classifier.state_dict()['1.weight'][0])}")

import torchvision
from torch import nn

#get the num out features (one for each class pizza, steak, sushi)

OUT_FEATURES = len(class_names)

#create EffNetB0 feature extractor
def create_effnetb0():
    #get the base model with pretrained weights and send to target device
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    #freeze tha base model layers
    for param in model.features.parameters():
        param.requires_grad=False

    #set the seeds
    set_seeds()

    #change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=OUT_FEATURES)
    ).to(device)

    #give the model a name
    model.name = 'effnetb0'
    print(f'created new {model.name} model')
    return model

#create an EffnetB2 feature extractor
def create_effnetb2():
    #get the base model with pretrained weights and sent to device
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights).to(device)

    #freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    #set the seeds
    set_seeds()

    #change the classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features=1408, out_features=OUT_FEATURES)
    ).to(device)

    #give the model name
    model.name = 'effnetb2'
    print(f'created new {model.name} model')
    return model


# effnetb0 = create_effnetb0()
# Get an output summary of the layers in our EffNetB0 feature extractor model
# summary(model=effnetb0,
#         input_size=(32, 3, 224, 224),
#         col_names=['input_size', 'output_size', 'num_params', 'trainable'],
#         col_width=20,
#         row_settings=['var_names'])

# effnetb2 = create_effnetb2()
# Get an output summary of the layers in our EffNetB0 feature extractor model
# summary(model=effnetb2,
#         input_size=(32, 3, 224, 224),
#         col_names=['input_size', 'output_size', 'num_params', 'trainable'],
#         col_width=20,
#         row_settings=['var_names'])

# Model	Total parameters (before freezing/changing head)	Total parameters (after freezing/changing head)	Total trainable parameters (after freezing/changing head)
# EfficientNetB0	5,288,548	4,011,391	3,843
# EfficientNetB2	9,109,994	7,705,221	4,227

# 7.6 Create experiments and set up training code
# We've prepared our data and prepared our models, the time has come to setup some experiments!
# We'll start by creating two lists and a dictionary:
# A list of the number of epochs we'd like to test ([5, 10])
# A list of the models we'd like to test (["effnetb0", "effnetb2"])
# A dictionary of the different training DataLoaders

#create epochs list
num_epochs = [5, 10]

# create models list (need to create a new model for each experiment)
models = ['effnetb0', 'effnetb2']

#create dataloaders dictionary for various dataloaders
train_dataloaders = {'data_10_percent':train_dataloader_10_percent,
                     'data_20_percent':train_dataloader_20_percent}

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
#             if model_name == 'effnetb0':
#                 model = create_effnetb0()  # creates a new model each time (important because we want each experiment to start from scratch)
#             else:
#                 model = create_effnetb2()  # creates a new model each time (important because we want each experiment to start from scratch)
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

# 9. Load in the best model and make predictions with it
#setup the best model filepath

best_model_path = 'models/07_effnetb2_data_20_percent_10_epochs.pth'

# Instantiate a new instance of EffNetB2 (to load the saved state_dict() to)
best_model = create_effnetb2()

#load the saved best model state_dict()
best_model.load_state_dict(torch.load(best_model_path))

#check the model file size
from pathlib import Path

#get the model size in bytes and then converts into megabytes
effnetb2_model_size = Path(best_model_path).stat().st_size // (1024 * 1024)
# print(f'EfficientNetB2 feature extractor model size {effnetb2_model_size} mb')

from going_modular.predictions import pred_and_plot_image

#get random list of 3 images from 20% test set
import random
num_images_to_plot = 3
test_image_path_list = list(Path(data_20_percent_path / 'test').glob('*/*.jpg'))
test_image_path_sample = random.sample(population=test_image_path_list,
                                       k=num_images_to_plot)

#iterate through random test image paths, make predictions on them and then plot
# for image_path in test_image_path_sample:
#     pred_and_plot_image(model=best_model,
#                         image_path=image_path,
#                         class_names=class_names,
#                         image_size=(224, 224))

# 9.1 Predict on a custom image with the best model
#download custom image
import requests

#setup custom image path
custom_image_path = Path('data/04-pizza-dad.jpeg')

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, 'wb') as f:
        request = requests.get('https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg')
        print(f'Downloading {custom_image_path}')
        f.write(request.content)
else:
    print('already exist')

#predict on custom image
pred_and_plot_image(model=best_model,
                    image_path=custom_image_path,
                    class_names=class_names)











