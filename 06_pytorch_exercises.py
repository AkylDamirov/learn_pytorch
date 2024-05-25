# Make predictions on the entire test dataset and plot a confusion matrix for the results of our
# model compared to the truth labels. Check out 03. PyTorch Computer Vision section 10 for ideas.

#import required libraries/code
import torch, torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torchinfo import summary

# Try to import the going_modular directory, download it from GitHub if it doesn't work
try:
    from going_modular import data_setup, engine
except ImportError:
    import os
    import subprocess
    print('Could not find going modular scripts.. downloading them from github')
    subprocess.run(['git', 'clone', 'https://github.com/mrdbourke/pytorch-deep-learning'], check=True)
    os.rename('pytorch-deep_learning/going_modular', 'going_modular')
    subprocess.run(['rm', '-rf', 'pytorch-deep-learning'], check=True)

    from going_modular import data_setup, engine

#setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import os
import zipfile
from pathlib import Path
import requests

#setup path to data folder

data_path = Path('data/')
image_path = data_path / 'pizza_steak_sushi'
# If the image folder doesn't exist, download it and prepare it...

if image_path.is_dir():
    print(f'{image_path} is exist')
else:
    print(f'Did not find {image_path} directory, creating one...')
    image_path.mkdir(parents=True, exist_ok=True)

    #download
    with open(data_path / 'pizza_steak_sushi.zip', 'wb') as f:
        request = requests.get('https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip')
        print('Downloading...')
        f.write(request.content)

    #unzip
    with zipfile.ZipFile(data_path / 'pizza_steak_sushi.zip', 'r') as zip_ref:
        print('Unzipping')
        zip_ref.extractall(image_path)

    #remove zipfile
    os.remove(data_path / 'pizza_steak_sushi.zip')


#setup dirs
train_dir = image_path / 'train'
test_dir = image_path / 'test'

#prepare data

# Create a transforms pipeline
simple_transform = transforms.Compose([
    transforms.Resize((224, 224)), # 1. Reshape all images to 224x224 (though some models may require different sizes)
    transforms.ToTensor(), # 2. Turn image values to between 0 & 1
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=simple_transform,
                                                                               batch_size=32)
#get and prepare pretrained model
# Get a set of pretrained model weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.features.parameters():
    param.requires_grad=False


# And we'll keep in_features=1280 for our Linear output layer but we'll change the out_features value to the length
# of our class_names (len(['pizza', 'steak', 'sushi']) = 3).
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280,
                    out_features=output_shape,
                    bias=True)).to(device)

#train model

#define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Note: We're only going to be training the parameters classifier here as all of the other parameters in our model have been frozen.
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# start the timer
from timeit import default_timer as timer
start_time = timer()

# #setup training and save the results

results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=5,
                       device=device)

end_time = timer()
print(f'Total training time {end_time-start_time:.3f} seconds')

#MAKE PREDICTIONS ON THE ENTIRE TEST DATASET WITH THE MODEL

from tqdm.auto import tqdm

#make predictions on the entire dataset
test_preds = []
model.eval()
with torch.inference_mode():
    #loop through batches in the test dataloaders
    for X,y in tqdm(test_dataloader):
        X,y = X.to(device), y.to(device)

        #pass the data through the model
        test_logits = model(X)

        #convert the pred logits to pred probs
        pred_probs = torch.softmax(test_logits, dim=1)

        #convert pred probs into labels
        pred_labels = torch.argmax(pred_probs, dim=1)

        #add the pred labels to test predict list
        test_preds.append(pred_labels)

#concatenate the test preds and put them on CPU
test_preds = torch.cat(test_preds).cpu()
# print(test_preds)

#make a confusion matrix with the test preds and the truth labels
#get the truth labels for test dataset
test_truth = torch.cat([y for X, y in test_dataloader])
# print(test_truth)

#libraries for confusion matrix
import subprocess
try:
    import torchmetrics
    import mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")
    assert int(mlxtend.__version__.split(".")[1]) >= 19, "mlxtend version should be 0.19.0 or higher"
except ImportError:
    subprocess.run(["pip3", "install", "-q", "torchmetrics", "-U", "mlxtend"])
    import mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")

#confusion matrix

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

#setup confusion matrix instance
confmat = ConfusionMatrix(task='multiclass', num_classes=len(class_names))
confmat_tensor = confmat(preds=test_preds,
                         target=test_truth)

#plot confusion matrix
# fig, ax = plot_confusion_matrix(
#     conf_mat=confmat_tensor.numpy(),
#     class_names=class_names,
#     figsize=(10, 7)
# )
# plt.show()


# 2. Get the "most wrong" of the predictions on the test dataset and plot the 5 "most wrong" images. You can do this by:
# Predicting across all of the test dataset, storing the labels and predicted probabilities.
# Sort the predictions by wrong prediction and then descending predicted probabilities, this will give you the wrong predictions with the highest prediction probabilities, in other words, the "most wrong".
# Plot the top 5 "most wrong" images, why do you think the model got these wrong?

#create dataFrame with sample, label, prediction,pred_porbs
#get all test data paths
from pathlib import Path
test_data_paths = list(Path(test_dir).glob('*/*.jpg'))
test_labels = [path.parent.stem for path in test_data_paths]

#create a function to return  list of dict with sample, label prediction and probs
def pred_and_store(test_paths, model, transform, class_names):
    test_pred_list = []
    for path in test_paths:
        #create empty dict to store info each sample
        pred_dict = {}

        #get sample bath
        pred_dict['image_path'] = path

        #get class name
        class_name = path.parent.stem
        pred_dict['class_name'] = class_name

        #get prediction and prediction probability
        from PIL import Image
        img = Image.open(path) #open image
        transformed_image = transform(img).unsqueeze(0)
        model.eval()
        with torch.inference_mode():
            pred_logit = model(transformed_image.to(device))
            pred_prob = torch.softmax(pred_logit, dim=1)
            pred_label = torch.argmax(pred_prob, dim=1)
            pred_class = class_names[pred_label.cpu()]

            #make sure it in cpu
            pred_dict['pred_prob'] = pred_prob.unsqueeze(0).max().cpu().item()
            pred_dict['pred_class'] = pred_class

        #does the pred match the true label?
        pred_dict['correct'] = class_name == pred_class

        #add dictionary to the list of preds
        test_pred_list.append(pred_dict)

    return test_pred_list

test_pred_dicts = pred_and_store(test_paths=test_data_paths,
               model=model,
               transform=simple_transform,
               class_names=class_names)

# print(test_pred_dicts[:5])

#turn the test_pred_dicts into a Dataframe
import pandas as pd

test_pred_df = pd.DataFrame(test_pred_dicts)
#sort dataframe by correct than by preb_prob
top_5_most_wrong = test_pred_df.sort_values(by=['correct', 'pred_prob'], ascending=[True, False]).head()

import torchvision
import matplotlib.pyplot as plt
#plot the top 5 most wrong images
# for row in top_5_most_wrong.iterrows():
#     row = row[1]
#     image_path = row[0]
#     true_label = row[1]
#     pred_prob = row[2]
#     pred_class = row[3]
#     #plot the image and varios details
#     img = torchvision.io.read_image(str(image_path)) #get image as a tensor
#     plt.figure()
#     plt.imshow(img.permute(1,2,0))
#     plt.title(f'True {true_label} | Pred: {pred_class} | Prob {pred_prob:.3f}')
#     plt.axis(False)
#     plt.show()







