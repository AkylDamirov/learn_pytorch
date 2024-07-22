#getting set up
import os
try:
    import torch
    import torchvision
    assert int(torch.__version__.split(".")[1]) >= 12, "torch version should be 1.12+"
    assert int(torchvision.__version__.split(".")[1]) >= 13, "torchvision version should be 0.13+"
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
except:
    print(f"[INFO] torch/torchvision versions not as required, installing nightly versions.")
    os.system('pip3 install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113')
    import torch
    import torchvision
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")


# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms


try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")
    os.system('pip install -q torchinfo')
    from torchinfo import summary


try:
    from going_modular import data_setup, engine
    from helper_functions import download_data, set_seeds, plot_loss_curves
except:
    # Get the going_modular scripts
    print("[INFO] Couldn't find going_modular or helper_functions scripts... downloading them from GitHub.")
    os.system('git clone https://github.com/mrdbourke/pytorch-deep-learning')
    os.system('mv pytorch-deep-learning/going_modular .')
    os.system('mv pytorch-deep-learning/helper_functions.py .') # get the helper_functions.py script
    os.system('rm -rf pytorch-deep-learning')
    from going_modular import data_setup, engine
    from helper_functions import download_data, set_seeds, plot_loss_curves

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#getting data
data_20_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
                                     destination="pizza_steak_sushi_20_percent")

train_dir = data_20_percent_path / 'train'
test_dir = data_20_percent_path / 'test'


# 3. Creating an EffNetB2 feature extractor
# 1. Setup pretrained EffNetB2 weights
# effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT

#2 get effnetb2 transfroms
# effnetb2_transforms = effnetb2_weights.transforms()

#3 setup pretrained model
# effnetb2 = torchvision.models.efficientnet_b2(weights=effnetb2_weights)

# 4.Freeze the base layers in the model(this will freeze all layers to begin with)
# for param in effnetb2.parameters():
#     param.requires_grad = False

# print(effnetb2.classifier)
# 5. Update the classifier head

# effnetb2.classifier = nn.Sequential(
#     nn.Dropout(p=0.3, inplace=True),
#     nn.Linear(in_features=1408,#keep in features the same
#               out_features=3) # change out_features to suit our number of classes
# )


# 3.1 Creating a function to make an EffNetB2 feature extractor
def create_effnetb2_model(num_classes=3, seed=42):
    # 1, 2, 3. Create EffNetB2 pretrained weights, transforms and model
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    #4 Freeze all layers in base model
    for param in model.parameters():
        param.requires_grad = False

    # 5. Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes)
    )

    return model, transforms

effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=3,seed=42)

#print summary
from torchinfo import summary

# summary(effnetb2,
#         input_size=(1, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])


# 3.2 Creating DataLoaders for EffNetB2
#set up dataloaders
train_dataloader_effnetb2, test_dataloader_effnetb2, class_name = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                                test_dir=test_dir,
                                                                                                transform=effnetb2_transforms,
                                                                                                batch_size=32)

# 3.3 Training EffNetB2 feature extractor
#setup optimizer
optimizer = torch.optim.Adam(params=effnetb2.parameters(), lr=1e-3)

#setup loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Set seeds for reproducibility and train the model
set_seeds()
# effnetb2_results = engine.train(model=effnetb2,
#                                 train_dataloader=train_dataloader_effnetb2,
#                                 test_dataloader=test_dataloader_effnetb2,
#                                 epochs=10,
#                                 optimizer=optimizer,
#                                 loss_fn=loss_fn,
#                                 device=device)



# 3.4 Inspecting EffNetB2 loss curves
from helper_functions import plot_loss_curves

# plot_loss_curves(effnetb2_results)


# 3.5 Saving EffNetB2 feature extractor

from going_modular import utils

#save the model
# utils.save_model(model=effnetb2,
#                  target_dir='models',
#                  model_name='09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth')


# 3.6 Checking the size of EffNetB2 feature extractor
from pathlib import Path

#get the model size and then convert it to megabytes
pretrained_effnetb2_model_size = Path("models/09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth").stat().st_size // (1024*1024) # division converts bytes to megabytes (roughly)

# Count number of parameters in EffNetB2\
effnetb2_total_params = sum(torch.numel(param) for param in effnetb2.parameters())
# print(effnetb2_total_params)

#create dictionary with effnetb2 statistics
# effnetb2_stats = {'test_loss':effnetb2_results['test_loss'][-1],
#                   'test_acc':effnetb2_results['test_acc'][-1],
#                   'number_of_parameters':effnetb2_total_params,
#                   'model size (MB)':pretrained_effnetb2_model_size
#                   }

# 4. Creating a ViT feature extractor
def create_vit_model(num_classes:3, seed=42):
    # Create ViT_B_16 pretrained weights, transforms and model
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.vit_b_16(weights=weights)

    #freeze all layers in model
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier head to suit our needs (this will be trainable)
    torch.manual_seed(42)
    model.heads = nn.Sequential(
        nn.Linear(in_features=768, #keep the same
                  out_features=num_classes)
    )

    return model, transforms

# Create ViT model and transforms
vit, vit_transforms = create_vit_model(num_classes=3, seed=42)
# summary(vit,
#         input_size=(1, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

# 4.1 Create DataLoaders for ViT
train_dataloader_vit, test_dataloader_vit, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                       test_dir=test_dir,
                                                                                       transform=vit_transforms,
                                                                                       batch_size=32)

# 4.2 Training ViT feature extractor
#setup optimizer
optimizer = torch.optim.Adam(params=vit.parameters())

#setup loss
loss_fn = torch.nn.CrossEntropyLoss()

# Train ViT model with seeds set for reproducibility
# set_seeds()
# vit_results = engine.train(model=vit,
#                            train_dataloader=train_dataloader_vit,
#                            test_dataloader=test_dataloader_vit,
#                            epochs=10,
#                            optimizer=optimizer,
#                            loss_fn=loss_fn,
#                            device=device)


# 4.3 Inspecting ViT loss curves
# plot_loss_curves(vit_results)


# 4.4 Saving ViT feature extractor
utils.save_model(model=vit,
                 target_dir='models',
                 model_name='09_pretrained_vit_feature_extractor_pizza_steak_sushi_20_percent.pth')








