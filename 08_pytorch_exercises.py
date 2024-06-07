# Replicate the ViT architecture we created with in-built PyTorch transformer layers.
# You'll want to look into replacing our TransformerEncoderBlock() class with torch.nn.TransformerEncoderLayer() (these contain the same layers as our custom blocks).
# You can stack torch.nn.TransformerEncoderLayer()'s on top of each other with torch.nn.TransformerEncoder().

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

#setup directories paths to train ans test images
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

# 4.5 Turning the ViT patch embedding layer into a PyTorch module
rand_image_tensor = torch.rand(32, 3, 224, 224)
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int=3,
                 patch_size: int=16,
                 embedding_dim: int=768):
        super().__init__()
        self.patch_size = patch_size
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
        assert image_resolution % self.patch_size == 0, f'Input image size must be divisible by patch size, image shape:{image_resolution}, patch size:{self.patch_size}'

        #perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        #make sure that output shape has right order
        return x_flattened.permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]


patch_embedding = PatchEmbedding(patch_size=16)
patch_embedding_output = patch_embedding(rand_image_tensor)
# print(f'Input shape: {rand_image_tensor.shape}')
# print(f'Output shape: {patch_embedding_output.shape}')

transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768,
                                                       nhead=12,
                                                       dim_feedforward=3072,
                                                       dropout=0.1,
                                                       activation="gelu",
                                                       batch_first=True,
                                                       norm_first=True)



#Stack transformer encoder layers on top of each other to make full transformer encoder
transformer_encoder = nn.TransformerEncoder(
    encoder_layer=transformer_encoder_layer,
    num_layers=12
)


print(transformer_encoder)
# summary(model=transformer_encoder,
#         input_size=patch_embedding_output.shape)





