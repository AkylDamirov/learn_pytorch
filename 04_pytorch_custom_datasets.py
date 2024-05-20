import torch
from torch import nn

#setup device-agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Get data
# Let's write some code to download the formatted data from GitHub.
import requests
import zipfile
from pathlib import Path

#setup path to data folder
data_path = Path('data/')
image_path = data_path / 'pizza_steak_sushi'

#if the image folder doesn't exist download it and prepare it...

# if image_path.is_dir():
#     print(f'{image_path} directory exists')
# else:
#     print(f'Did not find {image_path} directory creating one...')
#     image_path.mkdir(parents=True, exist_ok=True)
#
#     #download pizza, steack and sushi data
#     with open(data_path / 'pizza_steak_sushi.zip', 'wb') as f:
#         request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
#         print('Downloading...')
#         f.write(request.content)
#
#     #unzip
#     with zipfile.ZipFile(data_path / 'pizza_steak_sushi.zip', 'r') as zip_ref:
#         print('Unzipping')
#         zip_ref.extractall(image_path)

#2. Become one with the data (data preparation)
# We can inspect what's in our data directory by writing a small helper function to walk through each of the subdirectories and count the files present.
# To do so, we'll use Python's in-built os.walk().
import os

def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f'There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}.')

# walk_through_dir(image_path)

#setup train and testing paths
train_dir = image_path / 'train'
test_dir = image_path / 'test'

# Visualize an image
# Get all of the image paths using pathlib.Path.glob() to find all of the files ending in .jpg.
# Pick a random image path using Python's random.choice().
# Get the image class name using pathlib.Path.parent.stem.
# And since we're working with images, we'll open the random image path using PIL.Image.open() (PIL stands for Python Image Library).
# We'll then show the image and print some metadata.

import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#set seed
torch.manual_seed(42)

#get all images paths
image_path_list = list(image_path.glob('*/*/*.jpg'))

#get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem

#open image
img = Image.open(random_image_path)

# 5. Print metadata
# print(f"Random image path: {random_image_path}")
# print(f"Image class: {image_class}")
# print(f"Image height: {img.height}")
# print(f"Image width: {img.width}")
# Turn the image into an array
img_as_array = np.asarray(img)

# Plot the image with matplotlib
# plt.figure(figsize=(10, 7))
# plt.imshow(img_as_array)
# plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
# plt.axis(False)
# plt.show()

# 3. Transforming data
# Now what if we wanted to load our image data into PyTorch?
# Before we can use our image data with PyTorch we need to:
# Turn it into tensors (numerical representations of our images).
# Turn it into a torch.utils.data.Dataset and subsequently a torch.utils.data.DataLoader, we'll call these Dataset and DataLoader for short.
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 3.1 Transforming data with torchvision.transforms
# To get experience with torchvision.transforms, let's write a series of transform steps that:
# Resize the images using transforms.Resize()).
# Flip our images randomly on the horizontal using transforms.RandomHorizontalFlip()
# (this could be considered a form of data augmentation because it will artificially change our image data).
# Turn our images from a PIL image to a PyTorch tensor using transforms.ToTensor().
# We can compile all of these steps using torchvision.transforms.Compose().

#transform for image
data_transform = transforms.Compose([
    #resize to 64x64
    transforms.Resize(size=(64, 64)),
    #flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    #turn images into torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])
# Now we've got a composition of transforms, let's write a function to try them out on various images.
def plot_transformed_images(image_paths,transform, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(f)
            ax[0].set_title(f'Original \n size:{f.size} ')
            ax[0].axis('off')

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1,2,0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f'Transformed \n Size: {transformed_image.shape}')
            ax[1].axis('off')

            fig.suptitle(f'class {image_path.parent.stem}', fontsize=16)
            plt.show()


# plot_transformed_images(image_path_list, transform=data_transform, n=3)

# 4. Option 1: Loading Image Data Using ImageFolder
# Use ImageFolder to create dataset(s)
from torchvision import datasets

train_data = datasets.ImageFolder(root=train_dir, #target folder of images
                                  transform=data_transform,# transforms to perform on data (images)
                                  target_transform=None)# transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

# print(f'Train data \n {train_data}, \nTest data \n{test_data}')

#get class as a list
class_names = train_data.classes
# print(class_names)

#can also get class names as a dict
# class_dict = train_data.class_to_idx
# print(class_dict)

# We can index on our train_data and test_data Dataset's to find samples and their target labels.
img, label = train_data[0][0], train_data[0][1]
# print(f"Image tensor:\n{img}")
# print(f"Image shape: {img.shape}")
# print(f"Image datatype: {img.dtype}")
# print(f"Image label: {label}")
# print(f"Label datatype: {type(label)}")


# Our images are now in the form of a tensor (with shape [3, 64, 64]) and the labels are in the form of an integer relating to a specific class (as referenced by the class_to_idx attribute).
# How about we plot a single image tensor using matplotlib?
# We'll first have to to permute (rearrange the order of its dimensions) so it's compatible.
# Right now our image dimensions are in the format CHW (color channels, height, width) but matplotlib prefers HWC (height, width, color channels).

#rearrange the order of dimensions
img_permute = img.permute(1,2,0)

#print different shapes
# print(f"Original shape: {img.shape} -> [color_channels, height, width]")
# print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

#plot the image
# plt.figure(figsize=(10, 7))
# plt.imshow(img.permute(1,2,0))
# plt.axis('off')
# plt.title(class_names[label], fontsize=14)
# plt.show()

# 4.1 Turn loaded images into DataLoader's
#turn train and test Datasets into Dataloaders
from torch.utils.data import DataLoader
# if __name__ == '__main__':
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=1,# how many samples per batch?
                              num_workers=0,# how many subprocesses to use for data loading? (higher = more) #change (1)
                              shuffle=True)# shuffle the data?

test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=1,
                                 num_workers=0, #change (1)
                                 shuffle=False)

    # Now our data is iterable.

img, label = next(iter(train_dataloader))
    # Batch size will now be 1, try changing the batch_size parameter above and see what happens
    # print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    # print(f"Label shape: {label.shape}")

# 5. Option 2: Loading Image Data with a Custom Dataset
# To see this in action, let's work towards replicating torchvision.datasets.ImageFolder() by subclassing torch.utils.data.Dataset (the base class for all Dataset's in PyTorch).
# We'll start by importing the modules we need:
# Python's os for dealing with directories (our data is stored in directories).
# Python's pathlib for dealing with filepaths (each of our images has a unique filepath).
# torch for all things PyTorch.
# PIL's Image class for loading images.
# torch.utils.data.Dataset to subclass and create our own custom Dataset.
# torchvision.transforms to turn our images into tensors.
# Various types from Python's typing module to add type hints to our code

import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List

# 5.1 Creating a helper function to get class names
# Let's write a helper function capable of creating a list of class names and a dictionary of class names and their indexes given a directory path.

# Get the class names using os.scandir() to traverse a target directory (ideally the directory is in standard image classification format).
# Raise an error if the class names aren't found (if this happens, there might be something wrong with the directory structure).
# Turn the class names into a dictionary of numerical labels, one for each class.

#setup path for target directory
target_directory = train_dir
# print(f'Target directory: {target_directory}')

#get the class names from the target directory
class_names_found = sorted([entry.name for entry in list(os.scandir(image_path / 'train'))])
# print(f'class names found {class_names_found}')

#make function to find classes in target directory
def find_classes(directory):
    #get class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    #raise an error if class not found
    if not classes:
        raise FileNotFoundError(f'could not find classes in {directory}')

    #create dictionary of index labels
    class_to_idx = {cls_name: i for i,cls_name in enumerate(classes)}
    return classes, class_to_idx

# print(find_classes(train_dir))

# 5.2 Create a custom Dataset to replicate ImageFolder
# Now we're ready to build our own custom Dataset.
# We'll build one to replicate the functionality of torchvision.datasets.ImageFolder().

# Subclass torch.utils.data.Dataset.
# Initialize our subclass with a targ_dir parameter (the target data directory) and transform parameter (so we have the option to transform our data if needed).
# Create several attributes for paths (the paths of our target images), transform (the transforms we might like to use, this can be None), classes and class_to_idx (from our find_classes() function).
# Create a function to load images from file and return them, this could be using PIL or torchvision.io (for input/output of vision data).
# Overwrite the __len__ method of torch.utils.data.Dataset to return the number of samples in the Dataset, this is recommended but not required. This is so you can call len(Dataset).
# Overwrite the __getitem__ method of torch.utils.data.Dataset to return a single sample from the Dataset, this is required.

from torch.utils.data import Dataset

#subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    #initialize with targ_dir and transform (optional) parameter
    def __init__(self, targ_dir, transform=None):
        #create attributes
        #get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob('*/*.jpg'))
        #setup transforms
        self.transform = transform
        #create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    #make func to load images
    def load_image(self, index):
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = self.load_image(index)
        class_name = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        #transform if necassary
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx

# Before we test out our new ImageFolderCustom class, let's create some transforms to prepare our images.
#augment train data
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

# Don't augment test data, only reshape
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Let's turn our training images (contained in train_dir) and our testing images
# (contained in test_dir) into Dataset's using our own ImageFolderCustom class.
train_data_custom = ImageFolderCustom(targ_dir=train_dir, transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)

# print(train_data_custom, test_data_custom)

#Check for equality amongst our custom Dataset and ImageFolder Dataset
# print((len(train_data_custom) == len(train_data)) & (len(test_data_custom) == len(test_data)))
# print(train_data_custom.classes == train_data.classes)
# print(train_data_custom.class_to_idx == train_data.class_to_idx)

# 5.3 Create a function to display random images
# Let's create a helper function called display_random_images() that helps us visualize images in our Dataset's.
# Specifically, it'll:
# Take in a Dataset and a number of other parameters such as classes (the names of our target classes), the number of images to display (n) and a random seed.
# To prevent the display getting out of hand, we'll cap n at 10 images.
# Set the random seed for reproducible plots (if seed is set).
# Get a list of random sample indexes (we can use Python's random.sample() for this) to plot.
# Setup a matplotlib plot.
# Loop through the random sample indexes found in step 4 and plot them with matplotlib.

# 1. Take in a Dataset as well as a list of class names
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes,
                          n=10,
                          display_shape=True,
                          seed=None):
    #adjust display if n too long
    if n>10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")

    if seed:
        random.seed(seed)

    #get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    #setup plot
    plt.figure(figsize=(16, 8))

    #loop through samples and display random samples
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0],dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1,2,0)

        #plot adjusted samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis('off')
        if classes:
            title = f'Class {classes[targ_label]}'
            if display_shape:
                title = title + f'Shape {targ_image_adjust.shape}'
        plt.title(title)
    plt.show()

# Display random images from ImageFolder created Dataset
# display_random_images(train_data, n=5, classes=class_names, seed=None)

# And now with the Dataset we created with our own ImageFolderCustom
# display_random_images(train_data_custom, n=12, classes=class_names, seed=None)

# 5.4 Turn custom loaded images into DataLoader's
from torch.utils.data import DataLoader

train_dataloader_custom = DataLoader(dataset=train_data_custom,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=True)

test_dataloader_custom = DataLoader(dataset=test_data_custom,
                                    batch_size=1,
                                    num_workers=0,
                                    shuffle=False)

#get image and label from custom Dataloader
img_custom, label_custom = next(iter(train_dataloader_custom))

# 6. Other forms of transforms (data augmentation)
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31), #how intese
    transforms.ToTensor() # use ToTensor() last to get everything between 0 & 1
])

# Don't need to perform augmentation on the test data
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Beautiful, now we've got a training transform (with data augmentation) and test transform (without data augmentation).
# Let's test our data augmentation out!
 #get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

#plot random images
# plot_transformed_images(
#     image_paths=image_path_list,
#     transform=train_transforms,
#     n=3,
#     seed=None
# )

# 7. Model 0: TinyVGG without data augmentation
# To begin, we'll start with a simple transform, only resizing the images to (64, 64) and turning them into tensors.

# 7.1 Creating transforms and loading data for Model 0
#create simple transform
simple_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Excellent, now we've got a simple transform, let's:
# Load the data, turning each of our training and test folders first into a Dataset with torchvision.datasets.ImageFolder()
# Then into a DataLoader using torch.utils.data.DataLoader().
# We'll set the batch_size=32 and num_workers to as many CPUs on our machine (this will depend on what machine you're using).

#load and transform data
train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

#Turn data into Dataloader's
import os
from torch.utils.data import DataLoader

#setup batch size and number of workers
BATCH_SIZE = 32
# NUM_WORKERS = os.cpu_count()
# print(f'Creating Dataloaders with batch size {BATCH_SIZE} and {NUM_WORKERS} workers')

#create Dataloader's
train_dataloader_simple = DataLoader(train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=False) #change (True)

test_dataloader_simple = DataLoader(test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=0) #change (NUM_WORKERS)

# print(train_dataloader_simple, test_dataloader_simple)

# 7.2 Create TinyVGG model class
# In notebook 03, we used the TinyVGG model from the CNN Explainer website.
# Let's recreate the same model, except this time we'll be using color images instead of grayscale (in_channels=3 instead of in_channels=1 for RGB pixels).

class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16, out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3,#number of color channels
                  hidden_units=10,
                  output_shape=len(train_data.classes)).to(device)

# print(model_0)

# 7.3 Try a forward pass on a single image (to test the model)

# To do a forward pass on a single image, let's:

# Get a batch of images and labels from the DataLoader.
# Get a single image from the batch and unsqueeze() the image so it has a batch size of 1 (so its shape fits the model).
# Perform inference on a single image (making sure to send the image to the target device).
# Print out what's happening and convert the model's raw output logits to prediction probabilities with torch.softmax()
# (since we're working with multi-class data) and convert the prediction probabilities to prediction labels with torch.argmax().

#get batch of images and labels from the Dataloader
# if __name__ == '__main__':
img_batch, label_batch = next(iter(train_dataloader_simple))

        #get a single image from the batch and unsqueeze the image so its shape fits the model
img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
        # print(f'Single image shape {img_single.shape}\n')

        #perform forward pass on a single pass
model_0.eval()
with torch.inference_mode():
    pred = model_0(img_single.to(device))

    # 4. Print out what's happening and convert model logits -> pred probs -> pred label
    # print(f"Output logits:\n{pred}\n")
    # print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    # print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    # print(f"Actual label:\n{label_single}")



# 7.4 Use torchinfo to get an idea of the shapes going through our model
# Install torchinfo if it's not available, import it if it is
import subprocess
import sys

try:
    import torchinfo
except:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torchinfo'])
    import torchinfo

from torchinfo import summary
# summary(model_0, input_size=[1, 3, 64, 64]) # do a test pass through of an example input size


# 7.5 Create train & test loop functions
def train_step(model:torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    #put model in train mode
    model.train()

    #setup train loss and loss accuracy values
    train_loss, train_acc = 0,0

    #loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        #send data to target device
        X,y = X.to(device), y.to(device)

        #forward pass
        y_pred = model(X)

        #calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        #optimizer zero grad
        optimizer.zero_grad()

        #loss backward
        loss.backward()

        #optimizer step
        optimizer.step()

        #calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class==y).sum().item() / len(y_pred)

    #adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    #put model in eval mode
    model.eval()

    #setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    #turn on inference context manager
    with torch.inference_mode():
        #loop through Dataloader batches
        for bacth, (X, y) in enumerate(dataloader):
            #send data to target device
            X,y = X.to(device), y.to(device)

            #forward pass
            test_pred_logits = model(X)

            #calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            #calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

        #adjust metrics to get average loss and accuracy per batch
        test_loss =  test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc

# 7.6 Creating a train() function to combine train_step() and test_step()
from tqdm.auto import tqdm

#take various parameters required for training and test steps
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):

    #create empty results dictionary
    results = {'train_loss': [],
               'train_acc': [],
               'test_loss':[],
               'test_acc': []}

    #loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)

        #print out whats happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        #update results dictionary
        results['train_loss'].append(train_loss)
        results['train_acc'].append(test_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

    return results

# 7.7 Train and Evaluate Model 0
# Time to put our TinyVGG model, DataLoader's and train() function together to see if we can build a model capable of discerning between pizza, steak and sushi!

#set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#set number of epochs
NUM_EPOCHS = 5

#recreate an instance on TinyVGG
model_0 = TinyVGG(input_shape=3,  # number of color channels (3 for RGB)
                  hidden_units=10,
                  output_shape=len(train_data.classes)).to(device)

#setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

#start timer
from timeit import default_timer as timer
# start_timer = timer()

#train model_0
# if __name__ == '__main__':
#     global model_0_results        #change
model_0_results = train(model=model_0,
                        train_dataloader=train_dataloader_simple,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)

    # End the timer and print out how long it took
    # end_time = timer()
    # print(f'Total training time {end_time-start_timer:.3f} seconds')

    # 7.8 Plot the loss curves of Model 0
    # Check the model_0_results keys
    # print(model_0_results.keys())

# We'll need to extract each of these keys and turn them into a plot.
def plot_loss_curves(results):
    #get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    #get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    #figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    #setup a plot
    plt.figure(figsize=(15, 7))

    #plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()


    #plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show()

    # plot_loss_curves(model_0_results)


# 9. Model 1: TinyVGG with Data Augmentation
# This time, let's load in the data and use data augmentation to see if it improves our results in anyway.
# First, we'll compose a training transform to include transforms.TrivialAugmentWide() as well as resize and turn our images into tensors.

# 9.1 Create transform with data augmentation
# Create training transform with TrivialAugment
train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

# Create testing transform (no data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
# Now let's turn our images into Dataset's using torchvision.datasets.ImageFolder() and then into DataLoader's with torch.utils.data.DataLoader()
# 9.2 Create train and test Dataset's and DataLoader's
# We'll make sure the train Dataset uses the train_transform_trivial_augment and the test Dataset uses the test_transform

#turn image folders into Datasets
train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform_trivial_augment)
test_data_simple = datasets.ImageFolder(test_dir, transform=test_transform)

# print(train_data_augmented, test_data_simple)

#turn Datasets into Dataloaders
import os

BATCH_SIZE = 32
# NUM_WORKERS = os.cpu_count()

torch.manual_seed(42)
train_dataloader_augment = DataLoader(train_data_augmented,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      num_workers=0) #changed (NUM_WORKERS)

test_dataloader_simple = DataLoader(test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=0) #changed (NUM_WORKERS)

# print(train_dataloader_augment, test_dataloader_simple)
# 9.3 Construct and train Model 1

#create model_1 and send it to the target device
torch.manual_seed(42)
model_1 = TinyVGG(
    input_shape=3,
    hidden_units=10,
    output_shape=len(train_data_augmented.classes)
).to(device)

# Since we've already got functions for the training loop (train_step()) and testing loop (test_step()) and a function to put them together in train(), let's reuse those.
#set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#set number of epochs
NUM_EPOCHS = 5

#setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(),lr=0.001)

#start the timer
# start_timer = timer()

#train model_1
# if __name__ == '__main__':
#     global model_1_results
model_1_results = train(model=model_1,
                        train_dataloader=train_dataloader_augment,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)

    # end_timer = timer()
    # print(f'Total training time {end_timer-start_timer:.3f} seconds')

    # 9.4 plot loss curves of model_1
    # plot_loss_curves(model_1_results)

# 10. Compare model results
import pandas as pd
model_0_df = pd.DataFrame(model_0_results)
model_1_df = pd.DataFrame(model_1_results)
# print(model_0_df)

# And now we can write some plotting code using matplotlib to visualize the results of model_0 and model_1 together.
#setup a plot
# plt.figure(figsize=(15, 10))
#
# #get number of epochs
# epochs = range(len(model_0_df))
#
# #plot train loss
# plt.subplot(2,2,1)
# plt.plot(epochs, model_0_df['train_loss'], label='Model 0')
# plt.plot(epochs, model_1_df['train_loss'], label='Model 1')
# plt.title('Train Loss')
# plt.xlabel('Epochs')
# plt.legend()
#
# #plot test loss
# plt.subplot(2, 2, 2)
# plt.plot(epochs, model_0_df['test_loss'], label='Model 0')
# plt.plot(epochs, model_1_df['test_loss'], label='Model 1')
# plt.title('Test loss')
# plt.xlabel('Epochs')
# plt.legend()
#
# #plot train accuracy
# plt.subplot(2, 2, 3)
# plt.plot(epochs, model_0_df['train_acc'], label='Model 0')
# plt.plot(epochs, model_1_df['train_acc'], label='Model 1')
# plt.title('Train accuracy')
# plt.xlabel('Epochs')
# plt.legend()
#
# #plot test accuracy
# plt.subplot(2, 2, 4)
# plt.plot(epochs, model_0_df['test_acc'], label='Model 0')
# plt.plot(epochs, model_1_df['test_acc'], label='Model 1')
# plt.title('Test accuracy')
# plt.xlabel('Epochs')
# plt.legend()
#
# plt.show()

# 11. Make a prediction on a custom image
#Download custom image
import requests

#setup custom image path
custom_image_path = data_path / '04-pizza-dad.jpeg'

# Download the image if it doesn't already exist
# if not custom_image_path.is_file():
#     with open(custom_image_path, 'wb') as f:
#         request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
#         print(f'Downloading {custom_image_path}...')
#         f.write(request.content)
# else:
#     print(f'{custom_image_path} already exists, skipping download.')

# 11.1 Loading in a custom image with PyTorch
# Since we want to load in an image, we'll use torchvision.io.read_image().
# This method will read a JPEG or PNG image and turn it into a 3 dimensional RGB or grayscale torch.Tensor with values of datatype uint8 in range [0, 255].

import torchvision

#read in custom image
custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))

# print(f'Custom image tensor \n {custom_image_uint8}\n')
# print(f'Custom image shape {custom_image_uint8.shape}\n')
# print(f'Custom image dtype {custom_image_uint8.dtype}')

# we'll need to convert it to the same format as the data our model is trained on.
# Load in custom image and convert the tensor values to float32
custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

# Divide the image pixel values by 255 to get them between [0, 1]
custom_image = custom_image / 255

# print(f'Custom image tensor \n {custom_image}\n')
# print(f'Custom image shape {custom_image.shape}\n')
# print(f'Custom image dtype {custom_image.dtype}')

# 11.2 Predicting on custom images with a trained PyTorch model
#plot custom image
# plt.imshow(custom_image.permute(1, 2, 0)) # need to permute image dimensions from CHW -> HWC otherwise matplotlib will error
# plt.title(f'Image shape {custom_image.shape}')
# plt.axis(False)
# plt.show()

# Now how could we get our image to be the same size as the images our model was trained on?
# One way to do so is with torchvision.transforms.Resize().

#Create transform pipleine to resize image
custom_image_transform = transforms.Compose([
    transforms.Resize((64, 64))
])

#transform the target image
custom_image_transformed = custom_image_transform(custom_image)

# print(f'Original shape {custom_image.shape}')
# print(f'New shape {custom_image_transformed.shape}')

#predict custom image

model_1.eval()
with torch.inference_mode():
    #add an extra dimension to image
    custom_image_transformed_with_batch_size= custom_image_transformed.unsqueeze(dim=0)

    # print(f'Custom image transformed shape: {custom_image_transformed.shape}')
    # print(f'unsqueezed custom image shape: {custom_image_transformed_with_batch_size.shape} ')

    #make predictions on image with extra dimension
    custom_image_pred = model_1(custom_image_transformed.unsqueeze(dim=0).to(device))

# Print out prediction logits
# print(f'Prediction logits {custom_image_pred}')

# Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
# print(f'Prediction probabilities {custom_image_pred_probs}')

# Convert prediction probabilities -> prediction labels
custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
# print(f'Prediction label {custom_image_pred_labels}')

# We can convert it to a string class name prediction by indexing on the class_names list.
#find predicted label
custom_image_pred_class = class_names[custom_image_pred_label.cpu()]
# print(custom_image_pred_class)

# 11.3 Putting custom image prediction together: building a function
def pred_and_plot_image(model: torch.nn.Module,
                        image_path,
                        class_names,
                        transform=None,
                        device: torch.device = device):
    #load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    #divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255

    #transform if necessary
    if transform:
        target_image = transform(target_image)

    #make sure the model on the target device
    model.to(device)

    #turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        #add extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        #make prediction on image with an extra dimension and send it to target device
        target_image_pred = model(target_image.to(device))

    # Convert logits -> prediction probabilities(using torch.softmax() for multi -class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1,2,0))
    if class_names:
        title = f'Pred {class_names[target_image_pred_label.cpu()]} | Prob {target_image_pred_probs.max().cpu():.3f}'
    else:
        title = f'Pred {target_image_pred_label} | Prob {target_image_pred_probs.max().cpu():.3f}'
    plt.title(title)
    plt.axis(False)
    plt.show()

#pred on our custom image
# pred_and_plot_image(model=model_1,
#                     image_path=custom_image_path,
#                     class_names=class_names,
#                     transform=custom_image_transform,
#                     device=device)




