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
for epoch in tqdm(range(epochs)):
    # print(f'Epoch {epoch}\n----')
    #Training
    train_loss = 0
    #Add a loop to loop through training batches
    for batch, (X,y) in enumerate(train_dataloader):
        model_0.train()
        #forward pass
        y_pred = model_0(X)

        #calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss # accumulatively add up the loss per epoch

        #optimizer zero grad
        optimizer.zero_grad()

        #loss backward
        loss.backward()

        #optimizer step
        optimizer.step()

        # Print out how many samples have been seen
        # if batch % 400 == 0:
        #     print(f'Looked at {batch * len(X)} / {len(train_dataloader.dataset)} samples')

    #divide total train loss by length of train dataloader (average loss per batch per epoch)
    train_loss /= len(train_dataloader)

    #Testing
    # setup variables for accumulatively adding up loss accuracy
    test_loss, test_acc = 0,0
    model_0.eval()
    with torch.inference_mode():
        for X,y in test_dataloader:
            #forward pass
            test_pred = model_0(X)

            #calculate loss
            test_loss += loss_fn(test_pred, y)

            #calculate accuracy
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        # Calculations on test metrics need to happen inside torch.inference_mode()
        # Divide total test loss by length of test dataloader (per batch)
        test_loss /= len(test_dataloader)

        # Divide total accuracy by length of test dataloader (per batch)
        test_acc /= len(test_dataloader)

    #print out whats happening
    # print(f'\nTrain loss: {train_loss:.5f}, test loss {test_loss:.5f}, Test acc {test_acc:.2f}%\n')

#calculate the running time
train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                                           end=train_time_end_on_cpu,
                                           device=str(next(model_0.parameters()).device))

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
train_time_start_on_gpu = timer()

epochs = 3
for epoch in tqdm(range(epochs)):
    print(f'Epoch {epoch}\n----')
    train_step(data_loader=train_dataloader,
               model=model_1,
               loss_fn = loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn)

    test_step(data_loader=test_dataloader,
              model=model_1,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn)


train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start= train_time_start_on_cpu, end=train_time_end_on_gpu, device=device)


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








