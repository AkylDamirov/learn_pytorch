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
set_seeds()
results = train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=5,
                device=device)

# print(results)
# 5. View our model's results in TensorBoard







