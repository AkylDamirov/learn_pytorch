# 0. Computer vision libraries in PyTorch
# torchvision	Contains datasets, model architectures and image transformations often used for computer vision problems.
# torchvision.datasets	Here you'll find many example computer vision datasets for a range of problems from image classification, object detection, image captioning, video classification and more. It also contains a series of base classes for making custom datasets.
# torchvision.models	This module contains well-performing and commonly used computer vision model architectures implemented in PyTorch, you can use these with your own problems.
# torchvision.transforms	Often images need to be transformed (turned into numbers/processed/augmented) before being used with a model, common image transformations are found here.
# torch.utils.data.Dataset	Base dataset class for PyTorch.
# torch.utils.data.DataLoader	Creates a Python iterable over a dataset (created with torch.utils.data.Dataset).

import torch
import torch.nn
#torchvision
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

#matplotlib
import matplotlib as plt

# 1. Getting a dataset
# set up training data
train_data = datasets.FashionMNIST(
    root='data',#where to download data to
    train=True, #get training data
    download=True, #download data if doesnt exist
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None #you can transform labels as well
)

#set up testing data

test_data = datasets.FashionMNIST(
    root='data',
    train=False, #get test data
    download=True,
    transform=ToTensor()
)

image,label = train_data[0]
# print(image, label)

# 1.1 Input and output shapes of a computer vision model
# print(image.shape)

#see classes
class_names = train_data.classes
# print(class_names)
import matplotlib.pyplot as plt
# 1.2 Visualizing our data
# print(f'Image shape: {image.shape}')
# plt.imshow(image.squeeze())  # image shape is [1, 28, 28] (colour channels, height, width)
# plt.title(label)
# plt.show()

# We can turn the image into grayscale using the cmap parameter of plt.imshow().
# plt.imshow(image.squeeze(), cmap='gray')
# plt.title(class_names[label])
# plt.show()

#plot more images
# torch.manual_seed(42)
# fig = plt.figure(figsize=(9,9))
# rows, cols = 4,4
# for i in range(1, rows*cols+1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap='gray')
#     plt.title(class_names[label])
#     plt.axis(False)
#     plt.show()


#2. prepare dataloader
from torch.utils.data import DataLoader

#setup the batch size hyperparameter
BATCH_SIZE = 32

#turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data,# dataset to turn into iterable
                              batch_size=BATCH_SIZE,# how many samples per batch?
                              shuffle=True)#shuffle every epoch

test_dataloader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False) # don't necessarily have to shuffle the testing data

# Let's check out what we've created
# print(f"Dataloaders: {train_dataloader, test_dataloader}")
# print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
# print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")

# Check out what's inside the training dataloader
train_features_batch, train_labels_bath = next(iter(train_dataloader))
# print(train_features_batch.shape, train_labels_bath.shape)

# And we can see that the data remains unchanged by checking a single sample.
# torch.manual_seed(42)
# random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
# img, label = train_features_batch[random_idx], train_labels_bath[random_idx]
# plt.imshow(img.squeeze(), cmap='gray')
# plt.title(class_names[label])
# plt.axis('Off')
# plt.show()
# print(f'Image size {img.shape}')
# print(f'label {label}, label.size {label.shape}')

# Model 0: Build a baseline model
#create a flatten layer
import torch.nn as nn
flatten_model = nn.Flatten()# all nn modules function as a model (can do a forward pass)

#get single sample
x = train_features_batch[0]

#flatten the sample
output = flatten_model(x) # perform forward pass

# Print out what happened
# print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
# print(f"Shape after flattening: {output.shape} -> [color_channels, height*width]")

class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), #neural networks like their inputs in vector form
            nn.Linear(in_features=input_shape,out_features=hidden_units),# in_features = number of features in a data sample (784 pixels)
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)

# We've got a baseline model class we can use, now let's instantiate a model.
torch.manual_seed(42)

#need setup model with input_parameters
model_0 = FashionMNISTModelV0(input_shape=784,# one for every pixel (28x28)
                              hidden_units=10, #how many units in the hidden layer
                              output_shape=len(class_names)
)
# print(model_0)

# 3.1 Setup loss, optimizer and evaluation metrics
import requests
from pathlib import Path

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  # Note: you need the "raw" GitHub URL for this to work
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)


#import accuracy metric
from helper_functions import accuracy_fn # Note: could also use torchmetrics.Accuracy(task = 'multiclass', num_classes=len(class_names)).to(device)

loss_fn = nn.CrossEntropyLoss() # this is also called "criterion"/"cost function" in some places
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

# 3.2 Creating a function to time our experiments
# It's time to start training a model.
# But how about we do a little experiment while we train.
# I mean, let's make a timing function to measure the time it takes our model to train on CPU versus using a GPU.
# We'll train this model on the CPU but the next one on the GPU and see what happens.
# Our timing function will import the timeit.default_timer() function from the Python timeit module.

from timeit import default_timer as timer
def print_train_time(start, end, device):
    total_time = end-start
    print(f'Train time on {device}: {total_time:.3f} seconds')
    return total_time

# 3.3 Creating a training loop and training a model on batches of data
#import tqdm for progress bar
from tqdm.auto import tqdm

#set the seed and start timer
torch.manual_seed(42)
train_time_start_on_cpu = timer()

# Set the number of epochs (we'll keep this small for faster training times)
epochs = 3

#create training and testing loop
# for epoch in tqdm(range(epochs)):
#     # print(f'Epoch {epoch}\n----')
#     #Training
#     train_loss = 0
#     #Add a loop to loop through training batches
#     for batch, (X,y) in enumerate(train_dataloader):
#         model_0.train()
#         #forward pass
#         y_pred = model_0(X)
#
#         #calculate the loss
#         loss = loss_fn(y_pred, y)
#         train_loss += loss # accumulatively add up the loss per epoch
#
#         #optimizer zero grad
#         optimizer.zero_grad()
#
#         #loss backward
#         loss.backward()
#
#         #optimizer step
#         optimizer.step()
#
#         # Print out how many samples have been seen
#         # if batch % 400 == 0:
#         #     print(f'Looked at {batch * len(X)} / {len(train_dataloader.dataset)} samples')
#
#     #divide total train loss by length of train dataloader (average loss per batch per epoch)
#     train_loss /= len(train_dataloader)
#
#     #Testing
#     # setup variables for accumulatively adding up loss accuracy
#     test_loss, test_acc = 0,0
#     model_0.eval()
#     with torch.inference_mode():
#         for X,y in test_dataloader:
#             #forward pass
#             test_pred = model_0(X)
#
#             #calculate loss
#             test_loss += loss_fn(test_pred, y)
#
#             #calculate accuracy
#             test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
#
#         # Calculations on test metrics need to happen inside torch.inference_mode()
#         # Divide total test loss by length of test dataloader (per batch)
#         test_loss /= len(test_dataloader)
#
#         # Divide total accuracy by length of test dataloader (per batch)
#         test_acc /= len(test_dataloader)
#
#     #print out whats happening
#     # print(f'\nTrain loss: {train_loss:.5f}, test loss {test_loss:.5f}, Test acc {test_acc:.2f}%\n')
#
# #calculate the running time
# train_time_end_on_cpu = timer()
# total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
#                                            end=train_time_end_on_cpu,
#                                            device=str(next(model_0.parameters()).device))

# 4. Make predictions and get Model 0 results
# Since we're going to be building a few models, it's a good idea to write some code to evaluate them all in similar ways.
# Namely, let's create a function that takes in a trained model, a DataLoader, a loss function and an accuracy function.
# The function will use the model to make predictions on the data in the DataLoader and then we can evaluate those predictions
# using the loss function and accuracy function.

torch.manual_seed(42)
def eval_model(model, data_loader, loss_fn, accuracy_fn):
    loss, acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            #make predictions with model
            y_pred = model(X)

            #accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))# For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)

        #scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {'model_name':model.__class__.__name__,# only works when model was created with a class
            'model_loss':loss.item(),
            'model_acc': acc}

#calculate model 0 results on the test dataset

model_0_results = eval_model(model=model_0, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn)

# print(model_0_r
# 5. Setup device agnostic-code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 6. Model 1: Building a better model with non-linearity

# We'll do so by recreating a similar model to before, except this time
# we'll put non-linear functions (nn.ReLU()) in between each linear layer.
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), #flatten inputs into single vector
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)
model_1 = FashionMNISTModelV1(
    input_shape = 784,
    hidden_units = 10,
    output_shape = len(class_names)
).to(device)
# print(next(model_1.parameters()).device)

# 6.1 Setup loss, optimizer and evaluation metrics

from helper_functions import accuracy_fn
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

# 6.2 Functionizing training and test loops
def train_step(model: torch.nn.Module, data_loader, loss_fn, optimizer, accuracy_fn, device: torch.device=device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X,y) in enumerate(data_loader):
        #send data to gpu
        X,y = X.to(device), y.to(device)

        #forward pass
        y_pred = model(X)

        #calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc = accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))# Go from logits -> pred labels

        #optimizer zero grad
        optimizer.zero_grad()

        # loss backward
        loss.backward()

        #optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f'Train loss {train_loss:.5f}, Train accuracy {train_acc:.2f}%')

def test_step(data_loader, model, loss_fn, accuracy_fn, device: torch.device=device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            X,y = X.to(device), y.to(device)

            #forward pass
            test_pred = model(X)

            #calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        #adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f'Test loss {test_loss:.5f}, Test accuracy {test_acc:.2f}%')

torch.manual_seed(42)

#measure time
from timeit import default_timer as timer
# train_time_start_on_gpu = timer()
#
# epochs = 3
# for epoch in tqdm(range(epochs)):
#     print(f'Epoch {epoch}\n----')
#     train_step(data_loader=train_dataloader,
#                model=model_1,
#                loss_fn = loss_fn,
#                optimizer=optimizer,
#                accuracy_fn=accuracy_fn)
#
#     test_step(data_loader=test_dataloader,
#               model=model_1,
#               loss_fn=loss_fn,
#               accuracy_fn=accuracy_fn)
#
#
# train_time_end_on_gpu = timer()
# total_train_time_model_1 = print_train_time(start= train_time_start_on_cpu, end=train_time_end_on_gpu, device=device)


# Let's evaluate our trained model_1 using our eval_model() function and see how it went.
torch.manual_seed(42)

model_1_results = eval_model(model=model_1,
                             data_loader=test_dataloader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn)
#
# print(model_1_results)
# 7. Model 2: Building a Convolutional Neural Network (CNN)
# To do so, we'll leverage the nn.Conv2d() and nn.MaxPool2d() layers from torch.nn
class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,# how big is the square that's going over the image?
                      stride=1,#default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)# default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)
# print(model_2)

#7.1 Stepping through nn.Conv2d()
# We could start using our model above and see what happens but let's first step through the two new layers we've added:
# nn.Conv2d(), also known as a convolutional layer.
# nn.MaxPool2d(), also known as a max pooling layer.

# To test the layers out, let's create some toy data just like the data used on CNN Explainer.
torch.manual_seed(42)

# Create sample batch of random numbers with same size as image batch
images = torch.rand(size=(32, 3, 64, 64)) # [batch_size, color_channels, height, width]
test_image = images[0] #get a single image for testing
# print(f"Image batch shape: {images.shape} -> [batch_size, color_channels, height, width]")
# print(f"Single image shape: {test_image.shape} -> [color_channels, height, width]")
# print(f"Single image pixel values:\n{test_image}")

# Let's create an example nn.Conv2d() with various parameters:

# in_channels (int) - Number of channels in the input image.
# out_channels (int) - Number of channels produced by the convolution.
# kernel_size (int or tuple) - Size of the convolving kernel/filter.
# stride (int or tuple, optional) - How big of a step the convolving kernel takes at a time. Default: 1.
# padding (int, tuple, str) - Padding added to all four sides of input. Default: 0.

# Example of what happens when you change the hyperparameters of a nn.Conv2d() layer.
torch.manual_seed(42)
# Create a convolutional layer with same dimensions as TinyVGG
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=0)

#pass data through the convolutional layer
# print(conv_layer(test_image))
# 7.2 Stepping through nn.MaxPool2d()
# Now let's check out what happens when we move data through nn.MaxPool2d().
# Print out original image shape without and with unsqueezed dimension

# print(f'Test image original step {test_image.shape}')
# print(f'Test image with unsqueezed dimension {test_image.unsqueeze(dim=0).shape}')

#create a sample nn.MaxPoo2d() layer
max_pool_layer = nn.MaxPool2d(kernel_size=2)

#pass the data through just the conv_layer
test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
# print(f"Shape after going through conv_layer(): {test_image_through_conv.shape}")

# Pass data through the max pool layer
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
# print(f"Shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}")

# Notice the change in the shapes of what's happening in and out of a nn.MaxPool2d() layer.
# The kernel_size of the nn.MaxPool2d() layer will effects the size of the output shape.
# In our case, the shape halves from a 62x62 image to 31x31 image.
# Let's see this work with a smaller tensor.

# 7.3 Setup a loss function and optimizer for model_2
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

# 7.4 Training and testing model_2 using our training and test functions
# We'll use our train_step() and test_step() functions we created before.
torch.manual_seed(42)

#measure time
from timeit import default_timer as timer
train_time_start_model_2 = timer()

#train and test model
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f'Epoch {epoch}------')
    train_step(data_loader=train_dataloader,
               model=model_2,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device)

    test_step(data_loader=test_dataloader,
              model=model_2,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_2, end=train_time_end_model_2, device=device)

#get model_2 results
model_2_results = eval_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
# print(model_2_results)
# 8. Compare model results and training time
# model_0 - our baseline model with two nn.Linear() layers.
# model_1 - the same setup as our baseline model except with nn.ReLU() layers in between the nn.Linear() layers.
# model_2 - our first CNN model that mimics the TinyVGG architecture on the CNN Explainer website.

# Let's combine our model results dictionaries into a DataFrame and find out.
import pandas as pd
compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
# print(compare_results)

# Visualize our model results
# compare_results.set_index('model_name')['model_acc'].plot(kind='barh')
# plt.xlabel('accuracy (%)')
# plt.ylabel('model')
# plt.show()

# 9. Make and evaluate random predictions with best model
# Alright, we've compared our models to each other, let's further evaluate our best performing model, model_2.
# To do so, let's create a function make_predictions() where we can pass the model and some data for it to predict on.
def make_predictions(model:torch.nn.Module, data, device:torch.device=device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            #prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            #forward pass
            pred_logit = model(sample)

            #get prediction probability
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            #get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)

import random
random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# View the first test sample shape and label
# print(f'Test sample image shape {test_samples[0].shape}\n test sample label: {test_labels[0]} ({class_names[test_labels[0]]})')




# Make predictions on test samples with model 2
pred_probs = make_predictions(model=model_2, data=test_samples)
# print(pred_probs[:2])

# And now we can go from prediction probabilities to prediction labels by taking the torch.argmax()
# of the output of the torch.softmax() activation function.

# Turn the prediction probabilities into prediction labels by taking the argmax()
pred_classes = pred_probs.argmax(dim=1)

# print(pred_classes)

#plot predictions
plt.figure(figsize=(9,9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
    #create a subplot
    plt.subplot(nrows, ncols, i+1)

    #plot the target image
    plt.imshow(sample.squeeze(), cmap='gray')

    #find the prediction label
    pred_label = class_names[pred_classes[i]]

    # Get the truth label (in text form, e.g. "T-shirt")
    truth_label = class_names[test_labels[i]]

    #title
    title_text = f'Pred {pred_label}, truth {truth_label}'
    # Check for equality and change title colour accordingly
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c='g') #green if correct
    else:
        plt.title(title_text, fontsize=10, c='r') #red wrong

    # plt.axis(False)
    # plt.show()

# 10. Making a confusion matrix for further prediction evaluation
# To make a confusion matrix, we'll go through three steps:
#
# Make predictions with our trained model, model_2 (a confusion matrix compares predictions to true labels).
# Make a confusion matrix using torchmetrics.ConfusionMatrix.
# Plot the confusion matrix using mlxtend.plotting.plot_confusion_matrix().

from tqdm.auto import tqdm

#make predictions with trained model
y_preds = []
model_2.eval()
with torch.inference_mode():
    for X,y in tqdm(test_dataloader, desc='Making predictions'):
        #send data and targets to target device
        X,y = X.to(device), y.to(device)
        #forward pass
        y_logit = model_2(X)
        # Turn predictions from logits -> prediction probabilities -> predictions labels
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
        # Put predictions on CPU for evaluation
        y_preds.append(y_pred.cpu())

# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)

# import subprocess
# try:
#     import torchmetrics
#     import mlxtend
#     print(f"mlxtend version: {mlxtend.__version__}")
#     assert int(mlxtend.__version__.split(".")[1]) >= 19, "mlxtend version should be 0.19.0 or higher"
# except ImportError:
#     subprocess.run(["pip3", "install", "-q", "torchmetrics", "-U", "mlxtend"])
#     import mlxtend
#     print(f"mlxtend version: {mlxtend.__version__}")

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
# 2. Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor, target=test_data.targets)

#plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),  # matplotlib likes working with NumPy
    class_names=class_names,
    figsize=(10, 7)
)
# plt.show()






