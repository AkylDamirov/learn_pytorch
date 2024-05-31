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

# 4.1 Calculating patch embedding input and output shapes by hand
#create example values
height = 224 #H (training resolution is 224)
width = 224 # W
color_channels = 3 #C
patch_size = 16 #P

#calculate the N (number of patches)

number_of_patches = int((height * width) / patch_size **2)
# print(f'Number of patches (N) with image height (H={height}), width (W={width}) and patch size (P={patch_size}): {number_of_patches}')

#input shape (this is this of single image)
embedding_layer_input_shape = (height, width, color_channels)

#output shape
embedding_layer_output_shape = (number_of_patches, patch_size**2 * color_channels)

# print(f'Input shape (single 2D image) {embedding_layer_input_shape}')
# print(f'Output shape (single 2D image flattened into patches): {embedding_layer_output_shape}')

# 4.2 Turning a single image into patches
#view single image
# plt.imshow(image.permute(1,2,0)) #adjust matplotlib
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()

# Change image shape to be compatible with matplotlib (color_channels, height, width) -> (height, width, color_channels)
image_permuted = image.permute(1, 2, 0)

# Index to plot the top row of patched pixels
patch_size = 16
# plt.figure(figsize=(patch_size, patch_size))
# plt.imshow(image_permuted[:patch_size, :, :])
# plt.show()

# Now we've got the top row, let's turn it into patches.
#setup hyperparameters and make sure img_size and patch_size are compatible
img_size = 224
patch_size = 16
num_patches = img_size/patch_size
assert img_size % patch_size == 0, 'Image size must be divisible by patch size'
print(f'Number of patches per row: {num_patches}\n Patch size {patch_size} pixels x {patch_size} pixels')

#create series of subplots
# fig, axs = plt.subplots(nrows=1,
#                        ncols=img_size //patch_size, #one column for each patch
#                        figsize=(num_patches, num_patches),
#                        sharex=True,
#                        sharey=True
#                        )

#iterate through number of patches in the top row
# for i, patch in enumerate(range(0, img_size, patch_size)):
#     axs[i].imshow(image_permuted[:patch_size, patch:patch+patch_size, :]) # keep height index constant, alter the width index
#     axs[i].set_xlabel(i+1) # set the label
#     axs[i].set_xticks([])
#     axs[i].set_yticks([])
# plt.show()

# This time we'll iterate through the indexs for height and width and plot each patch as it's own subplot.
# Setup hyperparameters and make sure img_size and patch_size are compatible
img_size = 224
patch_size=16
num_patches = img_size/patch_size
assert img_size % patch_size == 0, 'Image size must be divisible by patch size'
print(f"Number of patches per row: {num_patches}\
        \nNumber of patches per column: {num_patches}\
        \nTotal patches: {num_patches*num_patches}\
        \nPatch size: {patch_size} pixels x {patch_size} pixels")

#create a series of subplots
# fig, axs = plt.subplots(nrows=img_size//patch_size,# need int not float
#                         ncols=img_size//patch_size,
#                         figsize=(patch_size, patch_size),
#                         sharex=True,
#                         sharey=True
#                         )

#loop through height and width of image
# for i, patch_height in enumerate(range(0, img_size, patch_size)): #iterate through height
#     for j, patch_width in enumerate(range(0, img_size, patch_size)): #iterate through width
#         #plot the permuted image patch (image_permuted -> (Height, Width, Color Channels))
#         axs[i, j].imshow(image_permuted[patch_height:patch_height + patch_size,  # iterate through height
#                          patch_width:patch_width + patch_size,  # iterate through width
#                          :])  # get all color channels
#
#         #setup labels information, remove the ticks for clarity and set labels to outside
#         axs[i, j].set_ylabel(i+1,
#                              rotation='horizontal',
#                              horizontalalignment='right',
#                              verticalalignment='center')
#
#         axs[i,j].set_xlabel(i+1)
#         axs[i,j].set_xticks([])
#         axs[i,j].set_yticks([])
#         axs[i,j].label_outer()
#
# #set super title
# fig.suptitle(f'{class_names[label]} -> Patchified', fontsize=16)
# plt.show()

# 4.3 Creating image patches with torch.nn.Conv2d()
from torch import nn

#set the patch size
patch_size = 16

#create Conv2d layer with hyperparameters from the VIT paper
conv2d = nn.Conv2d(in_channels=3, #number of colors channels
                   out_channels=768, # from Table 1: Hidden size D, this is the embedding size
                   kernel_size=patch_size, # could also use (patch_size, patch_size)
                   stride=patch_size,
                   padding=0
                   )

# Now we've got a convoluational layer, let's see what happens when we pass a single image through it.
#view a single image
# plt.imshow(image.permute(1, 2, 0))
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()









