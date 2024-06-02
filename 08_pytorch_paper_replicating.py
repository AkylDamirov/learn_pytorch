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
# print(f"Number of patches per row: {num_patches}\
#         \nNumber of patches per column: {num_patches}\
#         \nTotal patches: {num_patches*num_patches}\
#         \nPatch size: {patch_size} pixels x {patch_size} pixels")

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

#pass the image through the convolutional layer
image_out_of_conv = conv2d(image.unsqueeze(0)) # add a single batch dimension (height, width, color_channels) -> (batch, height, width, color_channels)
# print(image_out_of_conv.shape)

#plot random 5 convolutional features maps
import random
random_indexes = random.sample(range(0, 758), k=5)
# print(f'showing random convolutional features maps from indexes {random_indexes}')
#
# #create plot
# fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(12, 12))
#
# #plot random image features maps
# for i, idx in enumerate(random_indexes):
#     image_conv_feature_map = image_out_of_conv[:, idx, :, :] #index on the output tensor of the convolutional layer
#     axs[i].imshow(image_conv_feature_map.squeeze().detach().numpy())
#     axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
# plt.show()

#Create flatten layer
flatten = nn.Flatten(start_dim=2, # flatten feature_map_height (dimension 2)
                     end_dim=3) # flatten feature_map_width (dimension 3)

#view single image
# plt.imshow(image.permute(1, 2, 0))
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()
# print(f'original image shape {image.shape}')

#turn image into feature maps
image_out_of_conv = conv2d(image.unsqueeze(0)) # add batch dimension to avoid shape errors
# print(f'Image feature map shape: {image_out_of_conv.shape}')

#flatten the feature maps
image_out_of_conv_flattened = flatten(image_out_of_conv)
# print(f'Flattened image feature map shape: {image_out_of_conv_flattened.shape}')

# Get flattened image patch embeddings in right shape
image_out_of_conv_flattened_reshaped = image_out_of_conv_flattened.permute(0, 2, 1)  # [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]

#get a single flattened feature map
single_flattened_feature_map = image_out_of_conv_flattened_reshaped[:, :, 0] # index: (batch_size, number_of_patches, embedding_dimension)

#plot the flattened feature map visually
# plt.figure(figsize=(22, 22))
# plt.imshow(single_flattened_feature_map.detach().numpy())
# plt.title(f'Flattened feature map shape: {single_flattened_feature_map.shape}')
# plt.axis(False)
# plt.show()

# 4.5 Turning the ViT patch embedding layer into a PyTorch module
# Time to put everything we've done for creating the patch embedding into a single PyTorch layer.
# We can do so by subclassing nn.Module and creating a small PyTorch "model" to do all of the steps above.
# Specifically we'll:
# Create a class called PatchEmbedding which subclasses nn.Module (so it can be used a PyTorch layer).
# Initialize the class with the parameters in_channels=3, patch_size=16 (for ViT-Base) and embedding_dim=768 (this is  𝐷  for ViT-Base from Table 1).
# Create a layer to turn an image into patches using nn.Conv2d() (just like in 4.3 above).
# Create a layer to flatten the patch feature maps into a single dimension (just like in 4.4 above).
# Define a forward() method to take an input and pass it through the layers created in 3 and 4.
# Make sure the output shape reflects the required output shape of the ViT architecture ( 𝑁×(𝑃2⋅𝐶) ).

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int=3,
                 patch_size: int=16,
                 embedding_dim: int=768):
        super().__init__()
        #create layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        #create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2,# only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    def forward(self, x):
        #create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % patch_size == 0, f'Input image size must be divisible by patch size, image shape:{image_resolution}, patch size:{patch_size}'

        #perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        #make sure that output shape has right order
        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]

set_seeds()

#create an instance of patch embedding layer
patchify = PatchEmbedding(in_channels=3,
                          patch_size=16,
                          embedding_dim=768)

#pass the single image through
# print(f'Input image shape {image.unsqueeze(0).shape}')
patch_embedded_image = patchify(image.unsqueeze(0))# add an extra batch dimension on the 0th index, otherwise will error

# print(f'output patch embedding shape {patch_embedded_image.shape}')

# Let's now get a summary of our PatchEmbedding layer.
random_input_image = (1, 3, 224, 224)
random_input_image_error = (1, 3, 250, 250) # will error because image size is incompatible with patch_size

# # Get a summary of the input and outputs of PatchEmbedding
# summary(PatchEmbedding(),
#         input_size=random_input_image,
#         col_names=['input_size', 'output_size', 'num_params', 'trainable'],
#         col_width=20,
#         row_settings=['var_names'])

# 4.6 Creating the class token embedding
#get the batch size and embedding dimension
batch_size = patch_embedded_image.shape[0]
embedding_dimension = patch_embedded_image.shape[-1]

#create the class token embedding as a learable parameter that shares the same size as the embedding dimension (D)
class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension),# [batch_size, number_of_tokens, embedding_dimension]
                           requires_grad=True)

#show the first 10 examples of the class token
# print(class_token[:, :, :10])
#
# #class token shape
# print(f'Class token shape {class_token.shape} -> [batch_size, number_of_tokens, embedding_dimension]')


#add the class token embedding to the front of the patch embedding
patch_embedded_image_with_class_embedding = torch.cat((class_token, patch_embedded_image),
                                                      dim=1)#concat on first dimenssion


# 4.7 Creating the position embedding
# So let's make a learnable 1D embedding with torch.ones() to create  𝐄pos  .
#calculate N (number of patches)
number_of_patches = int((height * width) /patch_size**2)

#get embedding dimension
embedding_dimension = patch_embedded_image_with_class_embedding.shape[2]

#create learnable 1d position embedding
position_embedding = nn.Parameter(torch.ones(1, number_of_patches+1, embedding_dimension),
                                  requires_grad=True)#make sure its learnable

#show the first 10  sequences and 10 position embedding values and check the shape of position embedding
# print(position_embedding[:, :10, :10])
# print(f'position embedding shape {position_embedding.shape} -> [batch_size, number_of_patches, embedding_dimension]')

#add the position embedding to the patch and class token embedding
patch_and_position_embedding = patch_embedded_image_with_class_embedding + position_embedding

# 4.8 Putting it all together: from image to embedding
set_seeds()

#1. set patch size
patch_size = 16

#2. print shape of original image tensor and get the image dimensions
# print(f'image tensor shape {image.shape}')
height, width = image.shape[1], image.shape[2]

#3. get image tensor and add batch dimension
x = image.unsqueeze(0)
# print(f'Input image with batch dimension shape {x.shape}')

#4. Creating patch embedding layer
patch_embedding_layer = PatchEmbedding(in_channels=3,
                                       patch_size=patch_size,
                                       embedding_dim=768)

#5. pass image through patch embedding layer
patch_embedding = patch_embedding_layer(x)
# print(f'patch embedding shape: {patch_embedding.shape}')

#6. create class token embeddiung



batch_size = patch_embedding.shape[0]
embedding_dimension = patch_embedding.shape[-1]
class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension),
                           requires_grad=True)
# print(f'class token embedding shape {class_token.shape}')

#7. prepend class token embedding to patch embedding
patch_embedding_class_token = torch.cat((class_token, patch_embedding), dim=1)
# print(f'Patch embedding with class token shape: {patch_embedding_class_token.shape}')

#8. create postion embedding
number_of_patches = int((height * width) / patch_size**2)
position_embedding = nn.Parameter(torch.ones(1, number_of_patches+1, embedding_dimension),requires_grad=True)

#9. add position embedding to patch embedding with class token
patch_and_position_embedding = patch_embedding_class_token + position_embedding
# print(f'Patch and position embedding shape:{patch_and_position_embedding.shape}')

# 5. Equation 2: Multi-Head Attention (MSA)
# 5.3 Replicating Equation 2 with PyTorch layers
class MultiheadSelfAttentionBlock(nn.Module):
    #initialize a multi-head self-attention block ("MSA block")
    def __init__(self,
                 embedding_dim: int=768, #Hidden size D from the table 1 for Vit base
                 num_heads: int=12,#heads from table 1 from ViT base
                 attn_dropout:float=0):# does not look like the papaer uses any dropout in MSAblock
        super().__init__()

        #create Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        #create multi-head attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True) #does out batch dimension come first?

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, #query embeddings
                                             key=x,#key embeddings
                                             value=x,#value embeddings
                                             need_weights=False) #do we need the weights or just the layer outputs?
        return attn_output

#create an instance of MSABlock
multihead_self_attention_block = MultiheadSelfAttentionBlock(embedding_dim=768,#from table 1
                                                             num_heads=12)#from table 1

#pass patch and position image embedding through MSABlock
patched_image_through_msa_block = multihead_self_attention_block(patch_and_position_embedding)
# print(f'Input shape of MSA block: {patch_and_position_embedding.shape}')
# print(f'Output shape of MSA block {patched_image_through_msa_block.shape}')

# 6.2 Replicating Equation 3 with PyTorch layers
class MLPBlock(nn.Module):
    #initialize the class with hyperparameters from table 1 and table 3
    def __init__(self,
                 embedding_dim: int=768, # Hidden Size D from Table 1 for ViT-Base
                 mlp_size: int=3072, # MLP size from Table 1 for ViT-Base
                 dropout:float=0.1): # Dropout from Table 3 for ViT-Base
        super().__init__()

        #create norm layer
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        #create multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(), # "The MLP contains two layers with a GELU non-linearity
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),# take back to embedding_dim
            nn.Dropout(p=dropout)# "Dropout, when used, is applied after every dense layer.."
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

mlp_block = MLPBlock(embedding_dim=768, #from tabel 1
                     mlp_size=3072, #from table 1
                     dropout=0.1) #from table 3

#pass output of MSABlock through MLPBlock
patched_image_through_mlp_block = mlp_block(patched_image_through_msa_block)
# print(f'Input shape of MLP block {patched_image_through_msa_block.shape}')
# print(f'output shape of MLP block {patched_image_through_mlp_block.shape}')


# 7. Create the Transformer Encoder
# 7.1 Creating a Transformer Encoder by combining our custom made layers
class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int=768, #hidden size D from the table 1 and tabel 3
                 num_heads: int=12, #heads from tabel 1 for VIT base
                 mlp_size: int=3072, #MLP size from table 1 for VIT base
                 mlp_dropout: float=0.1, #amount for dropout for dense layer from table 3 for VIT base
                 attn_dropout: float=0): #amount for dropout for attention layers
        super().__init__()

        #create MSA block (equation 2)
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)

        #create MLP block (equation 3)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)

    def forward(self, x):
        #create residual connection for MSA block (add the input to the output)
        x = self.msa_block(x)+x

        #create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x)+x

        return x

#cretae an instance of TransformerEncoderBlock
transformer_encoder_block = TransformerEncoderBlock()

# summary(model=transformer_encoder_block,
#         input_size=(1, 197, 768),
#         col_names=['input_size', 'output_size', 'num_params', 'trainable'],
#         col_width=20,
#         row_settings=['var_names'])

# 7.2 Creating a Transformer Encoder with PyTorch's Transformer layers
# Create the same as above with torch.nn.TransformerEncoderLayer()
torch_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768, # Hidden size D from Table 1 for ViT-Base
                                                             nhead=12, # Heads from Table 1 for ViT-Base
                                                             dim_feedforward=3072,# MLP size from Table 1 for ViT-Base
                                                             dropout=0.1,# Amount of dropout for dense layers from Table 3 for ViT-Base
                                                             activation='gelu', # GELU non-linear activation
                                                             batch_first=True,# Do our batches come first?
                                                             norm_first=True # Normalize first or after MSA/MLP layers?
                                                             )

# print(torch_transformer_encoder_layer)


















