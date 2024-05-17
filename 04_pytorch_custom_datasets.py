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
if __name__ == '__main__':
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=1,# how many samples per batch?
                                  num_workers=1,# how many subprocesses to use for data loading? (higher = more)
                                  shuffle=True)# shuffle the data?

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=1,
                                 num_workers=1,
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
NUM_WORKERS = os.cpu_count()
# print(f'Creating Dataloaders with batch size {BATCH_SIZE} and {NUM_WORKERS} workers')

#create Dataloader's
train_dataloader_simple = DataLoader(train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=True)

test_dataloader_simple = DataLoader(test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS)

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
if __name__ == '__main__':
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
summary(model_0, input_size=[1, 3, 64, 64]) # do a test pass through of an example input size




