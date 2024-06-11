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

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download pizza, steak, sushi images from GitHub
image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")

# Setup directory paths to train and test images
train_dir = image_path / "train"
test_dir = image_path / "test"


# Create image size (from Table 3 in the ViT paper)
IMG_SIZE = 224

# Create transform pipeline manually
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Set the batch size
BATCH_SIZE = 32 # this is lower than the ViT paper but it's because we're starting small

# Create data loaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms, # use manually created transforms
    batch_size=BATCH_SIZE
)



# Get a batch of images
image_batch, label_batch = next(iter(train_dataloader))

# Get a single image from the batch
image, label = image_batch[0], label_batch[0]

#plot image with matplotlib
# plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()

rand_image_tensor = torch.randn(32, 3, 224, 224)


# 1. Create a class which subclasses nn.Module
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """

    # 2. Initialize the class with appropriate variables
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768):
        super().__init__()

        self.patch_size = patch_size

        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2,  # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # 6. Make sure the output shape has the right order
        return x_flattened.permute(0, 2,
                                   1)  # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]


patch_embedding = PatchEmbedding(patch_size=16)
patch_embedding_output = patch_embedding(rand_image_tensor)

# Hyperparameters from Table 1 and Table 3 for ViT-Base
transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768,
                                                       nhead=12,
                                                       dim_feedforward=3072,
                                                       dropout=0.1,
                                                       activation="gelu",
                                                       batch_first=True,
                                                       norm_first=True)


from torchinfo import summary

# summary(model=transformer_encoder_layer,
#         input_size=patch_embedding_output.shape)

transformer_encoder = nn.TransformerEncoder(
    encoder_layer=transformer_encoder_layer,
    num_layers=12
)

# summary(model=transformer_encoder,
#         input_size=patch_embedding_output.shape)
#

#PUT it all together and create VIT
class ViT(nn.Module):
    def __init__(self,
                 img_size=224, #from table 3
                 num_channels=3,
                 patch_size=16,
                 embedding_dim=768, #from table 1
                 dropout=0.1,
                 mlp_size=3072, #from table 1
                 num_transformer_layers=12, #from table 1
                 num_heads=12, #from table 1
                 num_classes=1000): #generic number of classes
        super().__init__()

        #assert image size is divisible by patch size
        assert img_size % patch_size == 0, 'Image must be divisible by patch size'

        #create patch embedding
        self.patch_embedding = PatchEmbedding(in_channels=num_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        #create class token
        self.class_token = nn.Parameter(torch.rand(1, 1, embedding_dim),
                                        requires_grad=True)

        #create positional embedding
        num_patches = (img_size * img_size) // patch_size**2  # N = HW/P^2
        self.positional_embedding = nn.Parameter(torch.rand(1, num_patches+1,embedding_dim))

        #create patch + position embedding dropout
        self.embedding_dropout = nn.Dropout(p=dropout)

        #create stack transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                                                  nhead=num_heads,
                                                                                                  dim_feedforward=mlp_size,
                                                                                                  activation='gelu',
                                                                                                  batch_first=True,
                                                                                                  norm_first=True), # Create a single Transformer Encoder Layer
                                                         num_layers=num_transformer_layers) #stack it N times

        #create MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
        #get some dimensions from x
        batch_size = x.shape[0]

        #create the patch embedding
        x = self.patch_embedding(x)

        #first, expand the class token across the batch size
        class_token = self.class_token.expand(batch_size, -1, -1)  # "-1" means infer the dimension

        #prepand the class token to the patch embedding
        x = torch.cat((class_token,x), dim=1)

        #add the positional embedding to patch embedding with class token
        x = self.positional_embedding + x

        #dropout on patch + positional embedding
        x = self.embedding_dropout(x)

        #pass embedding through transformer encoder stack
        x = self.transformer_encoder(x)

        #pass 0th index of x through MLP head
        x = self.mlp_head(x[:, 0])

        return x

demo_img = torch.rand(1, 3, 224, 224).to(device)

#create ViT
vit = ViT(num_classes=len(class_names)).to(device)
# print(vit(demo_img))


# summary(model=vit,
#         input_size=demo_img.shape)

embedding_dim = 768
class_token = nn.Parameter(torch.rand(1, 1, embedding_dim),
                           requires_grad=True)

batch_size = 32

class_token.expand(batch_size, -1, -1).shape # "-1" means to infer the dimension


patch_size = 16
img_size = 224
num_patches = (img_size*img_size) // patch_size**2
pos_embedding = nn.Parameter(torch.rand(1, num_patches+1, embedding_dim))
# print(pos_embedding.shape)


#2. Turn the custom ViT architecture we created into a Python script, for example, vit.py.
# You should be able to import an entire ViT model using something likefrom vit import ViT.
vit_model_code = """
import torch
import torch.nn as nn
class PatchEmbedding(nn.Module):
    # 2. Initialize the class with appropriate variables
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768):
        super().__init__()

        self.patch_size = patch_size

        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2,  # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # 6. Make sure the output shape has the right order
        return x_flattened.permute(0, 2,
                                   1)  # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]


class ViT(nn.Module):
    def __init__(self,
                 img_size=224, #from table 3
                 num_channels=3,
                 patch_size=16,
                 embedding_dim=768, #from table 1
                 dropout=0.1,
                 mlp_size=3072, #from table 1
                 num_transformer_layers=12, #from table 1
                 num_heads=12, #from table 1
                 num_classes=1000): #generic number of classes
        super().__init__()

        #assert image size is divisible by patch size
        assert img_size % patch_size == 0, 'Image must be divisible by patch size'

        #create patch embedding
        self.patch_embedding = PatchEmbedding(in_channels=num_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        #create class token
        self.class_token = nn.Parameter(torch.rand(1, 1, embedding_dim),
                                        requires_grad=True)

        #create positional embedding
        num_patches = (img_size * img_size) // patch_size**2  # N = HW/P^2
        self.positional_embedding = nn.Parameter(torch.rand(1, num_patches+1,embedding_dim))

        #create patch + position embedding dropout
        self.embedding_dropout = nn.Dropout(p=dropout)

        #create stack transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                                                  nhead=num_heads,
                                                                                                  dim_feedforward=mlp_size,
                                                                                                  activation='gelu',
                                                                                                  batch_first=True,
                                                                                                  norm_first=True), # Create a single Transformer Encoder Layer
                                                         num_layers=num_transformer_layers) #stack it N times

        #create MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
        #get some dimensions from x
        batch_size = x.shape[0]

        #create the patch embedding
        x = self.patch_embedding(x)

        #first, expand the class token across the batch size
        class_token = self.class_token.expand(batch_size, -1, -1)  # "-1" means infer the dimension

        #prepand the class token to the patch embedding
        x = torch.cat((class_token,x), dim=1)

        #add the positional embedding to patch embedding with class token
        x = self.positional_embedding + x

        #dropout on patch + positional embedding
        x = self.embedding_dropout(x)

        #pass embedding through transformer encoder stack
        x = self.transformer_encoder(x)

        #pass 0th index of x through MLP head
        x = self.mlp_head(x[:, 0])

        return x
"""

#write code into python file
# file_path = 'vit.py'
# with open(file_path, 'w') as file:
#     file.write(vit_model_code)

from vit import ViT

# imported_vit = ViT()
# summary(model=imported_vit,
#         input_size=(1, 3, 224, 224))
#

#3. Train a pretrained ViT feature extractor model (like the one we made in 08. PyTorch Paper Replicating section 10) on 20% of the pizza, steak and sushi data like the dataset we used in 07. PyTorch Experiment Tracking section 7.3.
# See how it performs compared to te EffNetB2 model we compared it to in 08. PyTorch Paper Replicating section 10.6.
set_seeds()

#create ViT feature extractor model
import torchvision

#download pretrained ViT weights and model
vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT #DEFAULT means best available
pretrained_vit = torchvision.models.vit_b_16(weights=vit_weights)

#freeze all layers in pretrained ViT model
for param in pretrained_vit.parameters():
    param.requires_grad = False

#update the pretrained ViT head
embedding_dim = 768 #ViT base
set_seeds()
pretrained_vit.heads = nn.Sequential(
    nn.LayerNorm(normalized_shape=embedding_dim),
    nn.Linear(in_features=embedding_dim,
              out_features=len(class_names))
)

#print summary
# summary(model=pretrained_vit,
#         input_size=(1, 3, 224, 224), # (batch_size, color_channels, height, width)
#         col_names=['input_size', 'output_size', 'num_params', 'trainable'],
#         col_width=20,
#         row_settings=['var_names'])

#get 20% of the data
data_20_percent_path = download_data(source='https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip',
                                     destination='pizza_steak_sushi_20_percent')

#setup train and test directories
train_dir_20_percent = data_20_percent_path / 'train'
#we dont need test 20%, 10% is enough

#preprocess the data
vit_transforms = vit_weights.transforms()
train_dataloader_20_percent, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir_20_percent,
                                                                                          test_dir=test_dir,
                                                                                          transform=vit_transforms,
                                                                                          batch_size=32)

# print(len(train_dataloader), len(train_dataloader_20_percent), len(test_dataloader))

#train a pretrained ViT feature extractor
from going_modular import engine

optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),
                             lr=1e-3)

loss_fn = torch.nn.CrossEntropyLoss()

set_seeds()
# pretrained_vit_results = engine.train(model=pretrained_vit,
#                                       train_dataloader=train_dataloader_20_percent,
#                                       test_dataloader=test_dataloader,
#                                       optimizer=optimizer,
#                                       loss_fn=loss_fn,
#                                       epochs=10,
#                                       device=device)

# 4. Try repeating the steps from excercise 3 but this time use the
# "ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1" pretrained weights from torchvision.models.vit_b_16()

# create ViT feature extractor model
import torchvision

#download pretrained ViT weights and model
vit_weights_swag = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 #get swag weights
pretrained_vit_swag = torchvision.models.vit_b_16(weights=vit_weights_swag)

#freeze all layers in pretrained ViT model
for param in pretrained_vit_swag.parameters():
    param.requires_grad = False

#update the pretrained ViT head
embedding_dim = 768
set_seeds()
pretrained_vit_swag.heads = nn.Sequential(
    nn.LayerNorm(normalized_shape=embedding_dim),
    nn.Linear(in_features=embedding_dim,
              out_features=len(class_names))
)

#check out transforms for pretrained ViT with Swag weights
vit_transforms_swag = vit_weights_swag.transforms()

# Get 20% of the data
data_20_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
                                     destination="pizza_steak_sushi_20_percent")

# Setup train and test directories
train_dir_20_percent = data_20_percent_path / "train"
# test_dir_20_percent = data_20_percent_path / "test" # don't need 20% test data as the model in 07. PyTorch Experiment Tracking section 7.3 tests on the 10% dataset not the 20%

# Preprocess the data
train_dataloader_20_percent, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir_20_percent,
                                                                                          test_dir=test_dir, # use 10% data for testing
                                                                                          transform=vit_transforms_swag,
                                                                                          batch_size=32)

#train pretrained ViT feature extractor with Swag weights

from going_modular import engine

optimizer = torch.optim.Adam(params=pretrained_vit_swag.parameters(),
                             lr=1e-3)

loss_fn = torch.nn.CrossEntropyLoss()

set_seeds()
# pretrained_vit_swag_results = engine.train(model=pretrained_vit_swag,
#                                            train_dataloader=train_dataloader_20_percent,
#                                            test_dataloader=test_dataloader,
#                                            optimizer=optimizer,
#                                            loss_fn=loss_fn,
#                                            epochs=10,
#                                            device=device)


#Bonus: get the most wrong examples from test dataset

#get all test data paths
from tqdm import tqdm
from pathlib import Path
test_data_paths = list(Path(test_dir).glob("*/*.jpg"))
test_labels = [path.parent.stem for path in test_data_paths]


# Create a function to return a list of dictionaries with sample, label, prediction, pred prob
def pred_and_store(test_paths, model, transform, class_names, device):
    test_pred_list = []
    for path in tqdm(test_paths):
        # Create empty dict to store info for each sample
        pred_dict = {}

        # Get sample path
        pred_dict["image_path"] = path

        # Get class name
        class_name = path.parent.stem
        pred_dict["class_name"] = class_name

        # Get prediction and prediction probability
        from PIL import Image
        img = Image.open(path)  # open image
        transformed_image = transform(img).unsqueeze(0)  # transform image and add batch dimension
        model.eval()
        with torch.inference_mode():
            pred_logit = model(transformed_image.to(device))
            pred_prob = torch.softmax(pred_logit, dim=1)
            pred_label = torch.argmax(pred_prob, dim=1)
            pred_class = class_names[pred_label.cpu()]

            # Make sure things in the dictionary are back on the CPU
            pred_dict["pred_prob"] = pred_prob.unsqueeze(0).max().cpu().item()
            pred_dict["pred_class"] = pred_class

        # Does the pred match the true label?
        pred_dict["correct"] = class_name == pred_class

        # print(pred_dict)
        # Add the dictionary to the list of preds
        test_pred_list.append(pred_dict)

    return test_pred_list


test_pred_dicts = pred_and_store(test_paths=test_data_paths,
                                 model=pretrained_vit_swag,
                                 transform=vit_transforms_swag,
                                 class_names=class_names,
                                 device=device)

# print(test_pred_dicts[:5])

#turn the test_pred dicts into Dataframe
import pandas as pd
test_pred_df = pd.DataFrame(test_pred_dicts)
# Sort DataFrame by correct then by pred_prob
top_5_most_wrong = test_pred_df.sort_values(by=["correct", "pred_prob"], ascending=[True, False]).head()
print(top_5_most_wrong)

#how many samples from the test dataset did our model get correct?
print(test_pred_df.correct.value_counts())

import torchvision
import matplotlib.pyplot as plt

# Plot the top 5 most wrong images
# for row in top_5_most_wrong.iterrows():
#     row = row[1]
#     image_path = row[0]
#     true_label = row[1]
#     pred_prob = row[2]
#     pred_class = row[3]
#     # Plot the image and various details
#     img = torchvision.io.read_image(str(image_path))  # get image as tensor
#     plt.figure()
#     plt.imshow(img.permute(1, 2, 0))  # matplotlib likes images in [height, width, color_channels]
#     plt.title(f"True: {true_label} | Pred: {pred_class} | Prob: {pred_prob:.3f}")
#     plt.axis(False);
#     plt.show()



