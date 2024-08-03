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

# effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=3,seed=42)

#print summary
from torchinfo import summary

# summary(effnetb2,
#         input_size=(1, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])


# 3.2 Creating DataLoaders for EffNetB2
#set up dataloaders
# train_dataloader_effnetb2, test_dataloader_effnetb2, class_names = data_setup.create_dataloaders(train_dir=train_dir,
#                                                                                                 test_dir=test_dir,
#                                                                                                 transform=effnetb2_transforms,
#                                                                                                 batch_size=32)

# 3.3 Training EffNetB2 feature extractor
#setup optimizer
# optimizer = torch.optim.Adam(params=effnetb2.parameters(), lr=1e-3)

#setup loss function
# loss_fn = torch.nn.CrossEntropyLoss()

# Set seeds for reproducibility and train the model
# set_seeds()
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
# pretrained_effnetb2_model_size = Path(
#     "demos/foodvision_mini2/09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth").stat().st_size // (1024 * 1024) # division converts bytes to megabytes (roughly)

# Count number of parameters in EffNetB2\
# effnetb2_total_params = sum(torch.numel(param) for param in effnetb2.parameters())
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
# vit, vit_transforms = create_vit_model(num_classes=3, seed=42)
# summary(vit,
#         input_size=(1, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

# 4.1 Create DataLoaders for ViT
# train_dataloader_vit, test_dataloader_vit, class_names = data_setup.create_dataloaders(train_dir=train_dir,
#                                                                                        test_dir=test_dir,
#                                                                                        transform=vit_transforms,
#                                                                                        batch_size=32)

# 4.2 Training ViT feature extractor
#setup optimizer
# optimizer = torch.optim.Adam(params=vit.parameters(), lr=1e-3)

#setup loss
# loss_fn = torch.nn.CrossEntropyLoss()

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
# utils.save_model(model=vit,
#                  target_dir='models',
#                  model_name='09_pretrained_vit_feature_extractor_pizza_steak_sushi_20_percent.pth')


# 4.5 Checking the size of ViT feature extractor
# Get the model size in bytes then convert to megabytes
# pretrained_vit_model_size = Path('models/09_pretrained_vit_feature_extractor_pizza_steak_sushi_20_percent.pth').stat().st_size // (1024*1024)
# print(f"Pretrained ViT feature extractor model size: {pretrained_vit_model_size} MB")


# 4.6 Collecting ViT feature extractor stats
# Count number of parameters in ViT
# vit_total_params = sum(torch.numel(param) for param in vit.parameters())
# print(vit_total_params)

# Create ViT statistics dictionary
# vit_stats = {'test_loss':vit_results['test_loss'][-1],
#              'test_acc':vit_results['test_acc'][-1],
#              'number_of_parameters': vit_total_params,
#              'model size (MB)': pretrained_vit_model_size}


# 5. Making predictions with our trained models and timing them
#get all test data paths
# print(f"[INFO] Finding all filepaths ending with '.jpg' in directory: {test_dir}")
test_data_paths = list(Path(test_dir).glob("*/*.jpg"))
# print(test_data_paths)


# 5.1 Creating a function to make predictions across the test dataset
import pathlib
from PIL import Image
from timeit import default_timer as timer
from tqdm.auto import tqdm
from typing import List, Dict

# 1. Create a function to return a list of dictionaries with sample, truth label, prediction, prediction probability and prediction time
def pred_and_store(paths: List[pathlib.Path],
                   model: torch.nn.Module,
                   transform: torchvision.transforms,
                   class_names: List[str],
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    # 2. Create an empty list to store prediction dictionaires
    pred_list = []

    # 3. Loop through target paths
    for path in tqdm(paths):
        # 4. Create empty dictionary to store prediction information for each sample
        pred_dict = {}

        # 5. Get the sample path and ground truth class name
        pred_dict['image_path'] = path
        class_name = path.parent.stem
        pred_dict['class_name'] = class_name

        # 6. Start the prediction timer
        start_time = timer()

        # 7. Open image path
        img = Image.open(path)

        # 8. Transform the image, add batch dimension and put image on target device
        transformed_image = transform(img).unsqueeze(0).to(device)

        # 9. Prepare model for inference by sending it to target device and turning on eval() mode
        model.to(device)
        model.eval()

        # 10. Get prediction probability, predicition label and prediction class
        with torch.inference_mode():
            pred_logit = model(transformed_image) # perform inference on target sample
            pred_prob = torch.softmax(pred_logit, dim=1) # turn logits into prediction probabilities
            pred_label = torch.argmax(pred_prob, dim=1) # turn prediction probabilities into prediction label
            pred_class = class_names[pred_label.cpu()] # hardcode prediction class to be on CPU

            # 11. Make sure things in the dictionary are on CPU (required for inspecting predictions later on)
            pred_dict['pred_prob'] = round(pred_prob.unsqueeze(0).max().cpu().item(), 4)
            pred_dict['pred_class'] = pred_class

            #12 end the timer and calculate time per pred
            end_time = timer()
            pred_dict['time_for_pred'] = round(end_time-start_time, 4)

        # 13. Does the pred match the true label?
        pred_dict['correct'] = class_name == pred_class

        # 14. Add the dictionary to the list of preds
        pred_list.append(pred_dict)

    # 15. Return list of prediction dictionaries
    return pred_list

# 5.2 Making and timing predictions with EffNetB2
# Make predictions across test dataset with EffNetB2
# effnetb2_test_pred_dicts = pred_and_store(paths=test_data_paths,
#                                           model=effnetb2,
#                                           transform=effnetb2_transforms,
#                                           class_names=class_names,
#                                           device='cpu')
# print(effnetb2_test_pred_dicts[:2])

# Turn the test_pred_dicts into a DataFrame
import pandas as pd

# effnetb2_test_pred_df = pd.DataFrame(effnetb2_test_pred_dicts)
# effnetb2_test_pred_df.head()

# Check number of correct predictions
# print(effnetb2_test_pred_df.correct.value_counts())

# Find the average time per prediction
# effnetb2_average_time_per_pred = round(effnetb2_test_pred_df.time_for_pred.mean(), 4)
# print(f"EffNetB2 average time per prediction: {effnetb2_average_time_per_pred} seconds")

# Add EffNetB2 average prediction time to stats dictionary
# effnetb2_stats['time_per_pred_cpu'] = effnetb2_average_time_per_pred

# 5.3 Making and timing predictions with ViT
# Make list of prediction dictionaries with ViT feature extractor model on test images
# vit_test_pred_dicts = pred_and_store(paths=test_data_paths,
#                                      model=vit,
#                                      transform=vit_transforms,
#                                      class_names=class_names,
#                                      device='cpu')

# Check the first couple of ViT predictions on the test dataset
# print(vit_test_pred_dicts[:2])

# Turn vit_test_pred_dicts into a DataFrame
# vit_test_pred_df = pd.DataFrame(vit_test_pred_dicts)
# vit_test_pred_df.head()

# Count the number of correct predictions
# print(vit_test_pred_df.correct.value_counts())

# Calculate average time per prediction for ViT model
# vit_average_time_per_pred = round(vit_test_pred_df.time_for_pred.mean(), 4)
# print(f"ViT average time per prediction: {vit_average_time_per_pred} seconds")

# Add average prediction time for ViT model on CPU
# vit_stats['time_per_pred_cpu'] = vit_average_time_per_pred
# print(vit_stats)

# 6. Comparing model results, prediction times and size
# Turn stat dictionaries into DataFrame
# df = pd.DataFrame([effnetb2_stats, vit_stats])
#
# #add column for model names
# df['model'] = ['EffNetB2', 'ViT']
#
# # Convert accuracy to percentages
# df['test_acc'] = round(df['test_acc']*100, 2)
#
# nan_columns = df.columns[df.isna().any()].tolist()
# print(f'Columns containing Nan values {nan_columns}')
#
# df = df[~df['model size (MB)'].isnull()]
# df[['model size (MB)']] = df[['model size (MB)']].astype(int)
#
# nan_columns = df.columns[df.isna().any()].tolist()
# if nan_columns:
#     print(f'After Columns containing Nan values {nan_columns}')
# else:
#     print('no Nan values')
# import numpy as np
# for stats in [effnetb2_stats, vit_stats]:
#     for key, value in stats.items():
#         if np.isnan(value) or np.isinf(value):
#             stats[key] = 0

# df['time_per_pred_cpu'] = pd.to_numeric(df['time_per_pred_cpu'], errors='coerce')
# df = df.dropna(subset=['time_per_pred_cpu'])
# df[['time_per_pred_cpu']] = df['time_per_pred_cpu'].astype(int)

# Compare ViT to EffNetB2 across different characteristics
# pd.DataFrame(data=(df.set_index("model").loc["ViT"] / df.set_index("model").loc["EffNetB2"]), # divide ViT statistics by EffNetB2 statistics
#              columns=["ViT to EffNetB2 ratios"]).T


# 6.1 Visualizing the speed vs. performance tradeoff
# 1. Create a plot from model comparison DataFrame
# fig, ax = plt.subplots(figsize=(12, 8))
# scatter = ax.scatter(data=df,
#                      x='time_per_pred_cpu',
#                      y='test_acc',
#                      c=['blue', 'orange'], #colours to use,
#                      s='model size (MB)'
#                      )
#
# # 2. Add titles, labels and customize fontsize for aesthetics
# ax.set_title('FoodFishion Mini Inference speed vs Perfomance', fontsize=18)
# ax.set_xlabel('Prediction time per image (seconds)', fontsize=14)
# ax.set_ylabel('test accuracy (%)', fontsize=14)
# ax.tick_params(axis='both', labelsize=12)
# ax.grid(True)
#
# # 3. Annotate with model names
# for index, row in df.iterrows():
#     ax.annotate(text=row['model'],
#                 xy=(row['time_per_pred_cpu']+0.0006, row['test_acc']+0.03),
#                 size=12)
#
#
# # 4. Create a legend based on model sizes
# handles, labels = scatter.legend_elements(prop='sizes', alpha=0.5)
# mobile_size_legend = ax.legend(handles,
#                                labels,
#                                loc='lower right',
#                                title='model size (MB)',
#                                fontsize=12)
#
# #save the figure
# os.makedirs('images', exist_ok=True)
# plt.savefig("images/09-foodvision-mini-inference-speed-vs-performance.jpg")

#show the figure
# plt.show()

# 7. Bringing FoodVision Mini to life by creating a Gradio demo
try:
    import gradio as gr
except:
    os.system('pip -q install gradio')
    import gradio as gr

# 7.2 Creating a function to map our inputs and outputs
# Put EffNetB2 on CPU
# effnetb2.to('cpu')

from typing import Tuple, Dict

# def predict(img) -> Tuple[Dict, float]:
#     """Transforms and performs a prediction on img and returns prediction and time taken.
#         """
#     #start the timer
#     start_time = timer()
#
#     # Transform the target image and add a batch dimension
#     img = effnetb2_transforms(img).unsqueeze(0)
#
#     # Put model into evaluation mode and turn on inference mode
#     effnetb2.eval()
#     with torch.inference_mode():
#         # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
#         pred_probs = torch.softmax(effnetb2(img), dim=1)
#
#     # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
#     pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
#
#     #calculate the prediction time
#     pred_time = round(timer() - start_time, 5)
#
#     # Return the prediction dictionary and prediction time
#     return pred_labels_and_probs, pred_time


import random
from PIL import Image

## Get a list of all test image filepaths
test_data_paths = list(Path(test_dir).glob('*/*.jpg'))

# Randomly select a test image path
# random_image_path = random.sample(test_data_paths, k=1)[0]

# Open the target image
# image = Image.open(random_image_path)
# print(f'Predicting on image at path: {random_image_path}')

# Predict on the target image and print out the outputs
# pred_dict, pred_time = predict(img=image)
# print(f"Prediction label and probability dictionary: \n{pred_dict}")
# print(f"Prediction time: {pred_time} seconds")

# 7.3 Creating a list of example images
# example_list = [[str(filepath)] for filepath in random.sample(test_data_paths, k=3)]

# 7.4 Building a Gradio interface
import gradio as gr

# title = 'FoodVision Mini'
# description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
# article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

# Create the Gradio demo
# demo = gr.Interface(fn=predict,
#                     inputs=gr.Image(type='pil'),
#                     outputs=[gr.Label(num_top_classes=3, label='Predictions'),
#                              gr.Number(label='Prediction time (s)')],
#                     examples=example_list,
#                     title=title,
#                     description=description,
#                     article=article)
#
# #launch the demo
# demo.launch(debug=False,
#             share=True)

# 8. Turning our FoodVision Mini Gradio Demo into a deployable app
# 8.3 Creating a demos folder to store our FoodVision Mini app files
import shutil
from pathlib import Path

# Create FoodVision mini demo path
foodvision_mini_demo_path = Path("demos/foodvision_mini2/")

# Remove files that might already exist there and create new directory
# if foodvision_mini_demo_path.exists():
#     shutil.rmtree(foodvision_mini_demo_path)
#     foodvision_mini_demo_path.mkdir(parents=True,
#                                     exist_ok=True)
#
# else:
#     # If the file doesn't exist, create it anyway
#     foodvision_mini_demo_path.mkdir(parents=True,
#                                     exist_ok=True)

# 8.4 Creating a folder of example images to use with our FoodVision Mini demo

# 1. Create an examples directory
foodvision_mini_examples_path = foodvision_mini_demo_path / 'examples'
# foodvision_mini_examples_path.mkdir(parents=True, exist_ok=True)

# 2. Collect three random test dataset image paths
foodvision_mini_examples = [Path('data/pizza_steak_sushi_20_percent/test/sushi/592799.jpg'),
                            Path('data/pizza_steak_sushi_20_percent/test/steak/3622237.jpg'),
                            Path('data/pizza_steak_sushi_20_percent/test/pizza/2582289.jpg')]

# 3. Copy the three random images to the examples directory
# for example in foodvision_mini_examples:
#     destination = foodvision_mini_examples_path / example.name
#     print(f"[INFO] Copying {example} to {destination}")
#     shutil.copy2(src=example, dst=destination)

# Get example filepaths in a list of lists
# example_list = [['examples/' + example] for example in os.listdir(foodvision_mini_examples_path)]
# print(example_list)

# 8.5 Moving our trained EffNetB2 model to our FoodVision Mini demo directory
# Create a source path for our target model
effnetb2_foodvision_mini_model_path = "models/09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth"

# Create a destination path for our target model
effnetb2_foodvision_mini_model_destination = foodvision_mini_demo_path / effnetb2_foodvision_mini_model_path.split('/')[1]

try:
    print(f"[INFO] Attempting to move {effnetb2_foodvision_mini_model_path} to {effnetb2_foodvision_mini_model_destination}")
    #move the model
    shutil.move(src=effnetb2_foodvision_mini_model_path,
                dst=effnetb2_foodvision_mini_model_destination)

    print('model move complete')

except:
    print(f"[INFO] No model found at {effnetb2_foodvision_mini_model_path}, perhaps its already been moved?")
    print(f"[INFO] Model exists at {effnetb2_foodvision_mini_model_destination}: {effnetb2_foodvision_mini_model_destination.exists()}")

# 8.6 Turning our EffNetB2 model into a Python script (model.py)
# file_path = foodvision_mini_demo_path / 'model.py'
# content = """
# def create_effnetb2_model(num_classes=3, seed=42):
#     # 1, 2, 3. Create EffNetB2 pretrained weights, transforms and model
#     weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
#     transforms = weights.transforms()
#     model = torchvision.models.efficientnet_b2(weights=weights)
#
#     #4 Freeze all layers in base model
#     for param in model.parameters():
#         param.requires_grad = False
#
#     # 5. Change classifier head with random seed for reproducibility
#     torch.manual_seed(seed)
#     model.classifier = nn.Sequential(
#         nn.Dropout(p=0.3, inplace=True),
#         nn.Linear(in_features=1408, out_features=num_classes)
#     )
#
#     return model, transforms
# """
#
# with open(file_path, 'w') as file:
#     file.write(content)

# 8.7 Turning our FoodVision Mini Gradio app into a Python script (app.py)
# file_path = foodvision_mini_demo_path / 'app.py'
# script_content = """
# import gradio as gr
# import os
# import torch
#
# from demos.foodvision_mini2.model import create_effnetb2_model
# from timeit import default_timer as timer
# from typing import Tuple, Dict
#
# # Setup class names
# class_names = ['pizza', 'steak', 'sushi']
#
# ### 2. Model and transforms preparation ###
# # Create EffNetB2 model
# effnetb2, effnetb2_transforms = create_effnetb2_model(len(class_names))
#
# # Load saved weights
# effnetb2.load_state_dict(
#     torch.load(
#         '09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth',
#         map_location=torch.device('cpu')
#     )
# )
#
# ### 3. Predict function ###
# # Create predict function
# def predict(img) -> Tuple[Dict, float]:
#     \"""Transforms and performs a prediction on img and returns prediction and time taken.\"""
#     # Start the timer
#     start_time = timer()
#
#     # Transform the target image and add a batch dimension
#     img = effnetb2_transforms(img).unsqueeze(0)
#
#     # Put model into evaluation mode and turn on inference mode
#     effnetb2.eval()
#     with torch.inference_mode():
#         # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
#         pred_probs = torch.softmax(effnetb2(img), dim=1)
#
#         # Create a prediction label and prediction probability dictionary for each prediction class
#         pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
#
#         # Calculate the prediction time
#         pred_time = round(timer() - start_time, 5)
#
#         # Return the prediction dictionary and prediction time
#         return pred_labels_and_probs, pred_time
#
# # Create title, description and article strings
# title = "FoodVision Mini 🍕🥩🍣"
# description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
# article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."
#
# # Create examples list from 'examples/' directory
# example_list = [['examples/' + example] for example in os.listdir('examples')]
#
# # Create gradio demo
# demo = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type='pil'),
#     outputs=[
#         gr.Label(num_top_classes=3, label='Predictions'),
#         gr.Number(label='Prediction time (s)')
#     ],
#     examples=example_list,
#     title=title,
#     description=description,
#     article=article
# )
#
# # Launch the demo
# demo.launch()
# """
#
#
# with open(file_path, 'w') as file:
#     file.write(script_content)

# file_path = foodvision_mini_demo_path / 'requirements.txt'
# content_requirements = """
# torch==1.12.0
# torchvision==0.13.0
# gradio==3.1.4
# """
#
# with open(file_path, 'w') as file:
#     file.write(content_requirements)

# 10. Creating FoodVision Big
# 10.1 Creating a model and transforms for FoodVision Big
# Create EffNetB2 model capable of fitting to 101 classes for Food101
effnetb2_food101, effnetb2_transforms = create_effnetb2_model(num_classes=101)

# Create Food101 training data transforms (only perform data augmentation on the training images)
food101_train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.TrivialAugmentWide(),
    effnetb2_transforms
])

# 10.2 Getting data for FoodVision Big
from torchvision import datasets

#setup data directory
data_dir = Path('data')

# Get training data (~750 images x 101 food classes)
train_data = datasets.Food101(root=data_dir,
                              split='train',
                              transform=food101_train_transforms, # perform data augmentation on training data
                              download=True)

test_data = datasets.Food101(root=data_dir,
                             split='test',
                             transform=effnetb2_transforms, # perform normal EffNetB2 transforms on test data
                             download=True)


# Get Food101 class names
food101_class_names = train_data.classes

# 10.3 Creating a subset of the Food101 dataset for faster experimenting

def split_dataset(dataset:torchvision.datasets, split_size:float=0.2, seed:int=42):
    # Create split lengths based on original dataset length
    length_1 = int(len(dataset)*split_size) #desired length
    length_2 = len(dataset)-length_1 #remaining length

    # Print out info
    print(f"[INFO] Splitting dataset of length {len(dataset)} into splits of size: {length_1} ({int(split_size * 100)}%), {length_2} ({int((1 - split_size) * 100)}%)")

    #create splits with given random seed
    random_split_1, random_split_2 = torch.utils.data.random_split(dataset,
                                                                   lengths=[length_1, length_2],
                                                                   generator=torch.manual_seed(seed))

    return random_split_1, random_split_2

#create training 20% split of food101
train_data_food101_20_percent, _ = split_dataset(dataset=train_data,
                                                 split_size=0.2)

#create testing 20% split of food101
test_data_food101_20_percent, _ = split_dataset(dataset=test_data,
                                                split_size=0.2)

# print(len(train_data_food101_20_percent), len(test_data_food101_20_percent))

# 10.4 Turning our Food101 datasets into DataLoaders
import os, torch

BATCH_SIZE = 32
NUM_WORKERS = 0

# Create Food101 20 percent training DataLoader
train_dataloader_food101_20_percent = torch.utils.data.DataLoader(train_data_food101_20_percent,
                                                                  batch_size=BATCH_SIZE,
                                                                  shuffle=True,
                                                                  num_workers=NUM_WORKERS)

# Create Food101 20 percent testing DataLoader
test_dataloader_food101_20_percent = torch.utils.data.DataLoader(test_data_food101_20_percent,
                                                                 batch_size=BATCH_SIZE,
                                                                 shuffle=False,
                                                                 num_workers=NUM_WORKERS
                                                                 )

# 10.5 Training FoodVision Big model
#setup optimizer
# optimizer = torch.optim.Adam(params=effnetb2_food101.parameters(),
#                              lr=1e-3)
#
# loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)# throw in a little label smoothing because so many classes

# Want to beat original Food101 paper with 20% of data, need 56.4%+ acc on test dataset
# set_seeds()
# effnetb2_food101_results = engine.train(effnetb2_food101,
#                                         train_dataloader=train_dataloader_food101_20_percent,
#                                         test_dataloader=test_dataloader_food101_20_percent,
#                                         optimizer=optimizer,
#                                         loss_fn=loss_fn,
#                                         epochs=5,
#                                         device=device)


# 10.6 Inspecting loss curves of FoodVision Big model
from helper_functions import plot_loss_curves

# Check out the loss curves for FoodVision Big
# plot_loss_curves(effnetb2_food101_results)

# 10.7 Saving and loading FoodVision Big

#create model path
# effnetb2_food101_model_path = '09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth'

#save foodision big model
# utils.save_model(model=effnetb2_food101,
#                  target_dir='models',
#                  model_name=effnetb2_food101_model_path)

# Create Food101 compatible EffNetB2 instance
# loaded_effnetb2_food101, effnetb2_transforms = create_effnetb2_model(num_classes=101)

# Load the saved model's state_dict()
# loaded_effnetb2_food101.load_state_dict(torch.load('09_pretrained_effnetb2_feature_extractor_food101_20_percent'))

# 10.8 Checking FoodVision Big model size
# Get the model size in bytes then convert to megabytes
# pretrained_effnetb2_food101_model_size = Path('models', effnetb2_food101_model_path).stat().st_size // (1024*1024) # division converts bytes to megabytes (roughly)
# print(f"Pretrained EffNetB2 feature extractor Food101 model size: {pretrained_effnetb2_food101_model_size} MB")

# 11. Turning our FoodVision Big model into a deployable app
#create foodvision big demo path
foodvision_big_demo_path = Path('demos/foodvision_big/')

#make foodvision big demo directory
foodvision_big_demo_path.mkdir(parents=True, exist_ok=True)

#make foodvision big demo examples directory
(foodvision_big_demo_path / 'examples').mkdir(parents=True, exist_ok=True)

# 11.1 Downloading an example image and moving it to the examples directory
import urllib.request

# Download and move an example image
# urllib.request.urlretrieve('https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg', '04-pizza-dad.jpeg')
# shutil.move('04-pizza-dad.jpeg', 'demos/foodvision_big2/examples/04-pizza-dad.jpg')
#
# # Move trained model to FoodVision Big demo folder (will error if model is already moved)
# try:
#     shutil.move('models/09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth', 'demos/foodvision_big2')
# except shutil.Error as e:
#     print(f'Error moving model {e}')


# 11.2 Saving Food101 class names to file (class_names.txt)
# Create path to Food101 class names
foodvision_big_class_names_path = foodvision_big_demo_path / 'class_names.txt'

# # Write Food101 class names list to file
# with open(foodvision_big_class_names_path, 'w') as f:
#     print(f"[INFO] Saving Food101 class names to {foodvision_big_class_names_path}")
#     f.write('\n'.join(food101_class_names))
#
# # 11.3 Turning our FoodVision Big model into a Python script (model.py)
# file_path = foodvision_big_demo_path / 'model.py'
# content = """
# import torch
# import torchvision
#
# from torch import nn
#
#
# def create_effnetb2_model(num_classes:int=3,
#                           seed:int=42):
#     # Create EffNetB2 pretrained weights, transforms and model
#     weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
#     transforms = weights.transforms()
#     model = torchvision.models.efficientnet_b2(weights=weights)
#
#     # Freeze all layers in base model
#     for param in model.parameters():
#         param.requires_grad = False
#
#     # Change classifier head with random seed for reproducibility
#     torch.manual_seed(seed)
#     model.classifier = nn.Sequential(
#         nn.Dropout(p=0.3, inplace=True),
#         nn.Linear(in_features=1408, out_features=num_classes),
#     )
#
#     return model, transforms
# """
# with open(file_path, 'w') as file:
#     file.write(content)
#
# file_path = foodvision_big_demo_path / 'app.py'
# content = """
# ### 1. Imports and class names setup ###
# import gradio as gr
# import os
# import torch
#
# from model import create_effnetb2_model
# from timeit import default_timer as timer
# from typing import Tuple, Dict
#
# #setup class names
# with open('class_names.txt', 'r') as f:
#     class_names = [food_name.strip for food_name in f.readline()]
#
# ### 2. Model and transforms preparation ###
#
# #create model
# effnetb2, effnetb2_transforms = create_effnetb2_model(
#     num_classes=101
# )
#
# #load saved weights
# effnetb2.load_state_dict(
#     torch.load(
#         f='09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth',
#         map_location=torch.device('cpu')
#     )
# )
#
# ### 3. Predict function ###
# #create predict function
# def predict(img):
#     #start timer
#     start_time = timer()
#
#     # Transform the target image and add a batch dimension
#     img = effnetb2_transforms(img).unsqueeze(0)
#
#     # Put model into evaluation mode and turn on inference mode
#     effnetb2.eval()
#     with torch.inference_mode():
#         # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
#         pred_probs = torch.softmax(effnetb2(img),dim=1)
#
#     # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
#     pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
#
#     # Calculate the prediction time
#     pred_time = (timer() - start_time, 5)
#
#     # Return the prediction dictionary and prediction time
#     return pred_labels_and_probs, pred_time
#
# ### 4. Gradio app ###
#
# # Create title, description and article strings
# title = "FoodVision Big 🍔👁"
# description = "An EfficientNetB2 feature extractor computer vision model to classify images of food into [101 different classes]"
# article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."
#
# # Create examples list from "examples/" directory
# example_list = [['examples/'+example] for example in os.listdir('examples')]
#
# # Create Gradio interface
# demo = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type='pil'),
#     outputs=[
#         gr.Label(num_top_classes=5, label="Predictions"),
#         gr.Number(label='Prediction time (s)')
#     ],
#     examples=example_list,
#     title=title,
#     description=description,
#     article=article
# )
#
# #launch the app ]
# demo.launch()
# """
# with open(file_path, 'w') as file:
#     file.write(content)
#
# file_path = foodvision_big_demo_path / 'requirements.txt'
# content_requirements = """
# torch==1.12.0
# torchvision==0.13.0
# gradio==3.1.4
# """
#
# with open(file_path, 'w') as file:
#     file.write(content_requirements)
#

# 11.7 Deploying our FoodVision Big app to HuggingFace Spaces
