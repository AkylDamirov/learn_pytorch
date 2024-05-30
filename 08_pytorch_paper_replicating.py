# 0. Getting setup
import torch, torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torchinfo import summary
from going_modular import data_setup, engine
from helper_functions import set_seeds, download_data, plot_loss_curves

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Get Data
#donwload pizza, sushi, steak
image_path = download_data(source='https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip',
                           destination='pizza_steak_sushi')

#setip directories paths to train ans test images
train_dir = image_path / 'train'
test_dir = image_path / 'test'

# 2. Create Datasets and DataLoaders
# 2.1 Prepare transforms for images
#create image size
IMG_SIZE = 224

#create transform pipeline manually
manual_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
# print(f'manually created transforms {manual_transform}')
# 2.2 Turn images into DataLoader's

#set the batch size
BATCH_SIZE = 32 # this is lower than the ViT paper but it's because we're starting small

#create dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transform,
    batch_size=BATCH_SIZE
)

# 2.3 Visualize a single image
#get batch of images
image_batch, label_batch = next(iter(train_dataloader))

#get the single image from batch

image, label = image_batch[0], label_batch[0]

#plot image with matplotlib
# plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()


# 3. Replicating the ViT paper: an overview
# 3.1 Inputs and outputs, layers and blocks







