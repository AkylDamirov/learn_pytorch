#0. Before we get into writing code, let's look at the general architecture of a classification neural network.

# Input layer shape (in_features)	Same as number of features (e.g. 5 for age, sex, height, weight, smoking status in heart disease prediction)	Same as binary classification
# Hidden layer(s)	Problem specific, minimum = 1, maximum = unlimited	Same as binary classification
# Neurons per hidden layer	Problem specific, generally 10 to 512	Same as binary classification
# Output layer shape (out_features)	1 (one class or the other)	1 per class (e.g. 3 for food, person or dog photo)
# Hidden layer activation	Usually ReLU (rectified linear unit) but can be many others	Same as binary classification
# Output activation	Sigmoid (torch.sigmoid in PyTorch)	Softmax (torch.softmax in PyTorch)
# Loss function	Binary crossentropy (torch.nn.BCELoss in PyTorch)	Cross entropy (torch.nn.CrossEntropyLoss in PyTorch)
# Optimizer	SGD (stochastic gradient descent), Adam (see torch.optim for more options)	Same as binary classification

#1. make classification data and get it ready
# We'll use the make_circles() method from Scikit-Learn to generate two circles with different coloured dots.

from sklearn.datasets import make_circles
import torch
# make 1000 samples
n_samples = 1000

#create circles

X,y = make_circles(n_samples,
                   noise=0.03, #a little bit noise to the dots
                   random_state=42) # keep random state so we get the same values

# print(f'First 5 x features:\n{X[:5]}')
# print(f'\nFirst 5 y labels:\n{y[:5]}')

# Looks like there's two X values per one y value.
# Let's keep following the data explorer's motto of visualize, visualize, visualize and put them into a pandas DataFrame.

# Make Dataframe of circle data
import pandas as pd
circles = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'label':y})
# print(circles.head(10))
# Check different labels
# print(circles.label.value_counts())
# Let's plot them.
# Visualize with a plot
import matplotlib.pyplot as plt

# plt.scatter(x=X[:, 0],
#             y=X[:, 1],
#             c=y,
#             cmap=plt.cm.RdYlBu)
# plt.show()

# 1.2 Turn data into tensors and create train and test splits
#turn data into tensors
#otherwise this causes issues with conputaions later on

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
#view the first five samples
# print(X[:5], y[:5])

# Now our data is in tensor format, let's split it into training and test sets.
# To do so, let's use the helpful function train_test_split() from Scikit-Learn.
# We'll use test_size=0.2 (80% training, 20% testing) and because the split happens randomly across the data,
# let's use random_state=42 so the split is reproducible.

#split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,#20% test 80% train
                                                    random_state=42)#make the random split reproducible

# print(len(X_train), len(X_test), len(y_train), len(y_test))

# 2. Building model
import torch.nn as nn

#make device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Construct a model class that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        #Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer1 = nn.Linear(in_features=2, out_features=5) #takes in 2 features (X), produces 5 features
        self.layer2 = nn.Linear(in_features=5, out_features=1)#takes in 5 features, produces 1 feature (y)

    def forward(self, x):
        return self.layer2(self.layer1(x))# computation goes through layer_1 first then the output of layer_1 goes through layer_2

#create an instance of the model and send it to target device
model_0 = CircleModelV0().to(device)

#you can do the same using nn.Sequential
# model_0 = nn.Sequential(
#     nn.Linear(in_features=2, out_features=5),
#     nn.Linear(in_features=5, out_features=1)
# ).to(device)


# Make predictions with the model
# untrained_preds = model_0(X_test.to(device))
# print(f"Length of predictions: {len(untrained_preds)}, Shape: {untrained_preds.shape}")
# print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
# print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
# print(f"\nFirst 10 test labels:\n{y_test[:10]}")

# 2.1 set up loss func and optimizer
# Stochastic Gradient Descent (SGD) optimizer	Classification, regression, many others.	torch.optim.SGD()
# Adam Optimizer	Classification, regression, many others.	torch.optim.Adam()
# Binary cross entropy loss	Binary classification	torch.nn.BCELossWithLogits or torch.nn.BCELoss
# Cross entropy loss	Mutli-class classification	torch.nn.CrossEntropyLoss
# Mean absolute error (MAE) or L1 Loss	Regression	torch.nn.L1Loss
# Mean squared error (MSE) or L2 Loss	Regression	torch.nn.MSELoss

#create a loss function
loss_fn = nn.BCEWithLogitsLoss() #BCEWithLogitsLoss = sigmoid built in
#create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()#torch.eq calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

#Train
# View the frist 5 outputs of the forward pass on the test data
y_logits = model_0(X_test.to(device))[:5]
# Use sigmoid on model logits
y_pred_probs = torch.sigmoid(y_logits)

#find predicted labels
y_preds = torch.round(y_pred_probs)

#in full
y_pred_lables = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

#check for equality
# print(torch.eq(y_preds.squeeze(), y_pred_lables.squeeze()))

#get rid of extra dimension
y_preds.squeeze()

# 3.2 building training and testing loop
torch.manual_seed(42)

#set number of epochs
epochs = 100

#put data into target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

#Build training and evaluting loop
for epoch in range(epochs):
    #Training
    model_0.train()
    #forward pass (model outputs raw logits)
    y_logits = model_0(X_train).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labs
    #calculate loss accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    #optimizer zero grad
    optimizer.zero_grad()
    #loss backwards
    loss.backward()
    #optimizer step
    optimizer.step()

    #Testing
    model_0.eval()
    with torch.inference_mode():
        #forward pass
        test_logits = model_0(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        #calculate loss/accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

    # if epoch % 10 == 0:
    #     print(f'Epoch {epoch}, Loss: {loss:.5f}, Accuracy: {acc:.2f}%, Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%')


#Make predictions and evalute the model
# Let's make a plot of our model's predictions, the data it's trying to predict on and the decision boundary it's creating for
#whether something is class 0 or class 1.
# To do so, we'll write some code to download and import the helper_functions.py script from the Learn PyTorch for Deep Learning repo.
# It contains a helpful function called plot_decision_boundary() which creates a NumPy meshgrid to visually plot the different points where
# our model is predicting certain classes.

import requests
from pathlib import Path

# Download helper functions from Learn PyTorch repo (if not already downloaded)
# if Path("helper_functions.py").is_file():
#   print("helper_functions.py already exists, skipping download")
# else:
#   print("Downloading helper_functions.py")
#   request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
#   with open("helper_functions.py", "wb") as f:
#     f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

#plot the decision boundaries for training and test sets
# plt.figure(figsize=(12,6))
# plt.subplot(1, 2, 1)
# plt.title('Train')
# plot_decision_boundary(model_0, X_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title('Test')
# plot_decision_boundary(model_0, X_test, y_test)
# plt.show()

#Improving a model (from a model perspective)
# Let's try to fix our model's underfitting problem.
# Focusing specifically on the model (not the data), there are a few ways we could do this.

# Add more layers	Each layer potentially increases the learning capabilities of the model with each layer being able to learn some kind of new pattern in the data, more layers is often referred to as making your neural network deeper.
# Add more hidden units	Similar to the above, more hidden units per layer means a potential increase in learning capabilities of the model, more hidden units is often referred to as making your neural network wider.
# Fitting for longer (more epochs)	Your model might learn more if it had more opportunities to look at the data.
# Changing the activation functions	Some data just can't be fit with only straight lines (like what we've seen), using non-linear activation functions can help with this (hint, hint).
# Change the learning rate	Less model specific, but still related, the learning rate of the optimizer decides how much a model should change its parameters each step, too much and the model overcorrects, too little and it doesn't learn enough.
# Change the loss function	Again, less model specific but still important, different problems require different loss functions. For example, a binary cross entropy loss function won't work with a multi-class classification problem.
# Use transfer learning	Take a pretrained model from a problem domain similar to yours and adjust it to your own problem. We cover transfer learning in notebook 06.

#Let's see what happens if we add an extra layer to our model, fit for longer (epochs=1000 instead of epochs=100)
# and increase the number of hidden units from 5 to 10.
# We'll follow the same steps we did above but with a few changed hyperparameters.

class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)#extra layer
        self.layer3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

model_1 = CircleModelV1().to(device)
# print(model_1)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)

torch.manual_seed(42)

epochs = 1000 # Train for longer

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    ### Training
    # 1. Forward pass
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits)) # logits -> predicition probabilities -> prediction labels

    # 2. Calculate loss/accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_1.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening every 10 epochs
    # if epoch % 100 == 0:
    #     print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

#plot decision boundaries for training and test sets
# plt.figure(figsize=(12,6))
# plt.subplot(1, 2, 1)
# plt.title('Train')
# plot_decision_boundary(model_1, X_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title('Test')
# plot_decision_boundary(model_1, X_test, y_test)
# plt.show()

#preparing data to see if our model can model a straight line
# Let's create some linear data to see if our model's able to model it and we're not just using a model that can't learn anything.
#create some data
weight = 0.7
bias = 0.3

#create data
X_regression = torch.arange(0, 1, 0.01).unsqueeze(dim=1)
y_regression = weight * X_regression + bias

# print(len(X_regression))
# print(X_regression[:5], y_regression[:5])

#lets split our data into training and test sets
train_split = int(0.8 * len(X_regression))
X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

# Check the lengths of each split
# print(len(X_train_regression),
#     len(y_train_regression),
#     len(X_test_regression),
#     len(y_test_regression))

# plot_predictions(train_data=X_train_regression,
#                  train_labels=y_train_regression,
#                  test_data=X_test_regression,
#                  test_labels=y_test_regression)
# plt.show()

# 5.2 Adjusting model_1 to fit a straight line
# Now we've got some data, let's recreate model_1 but with a loss function suited to our regression data.
#Same arhitecture as model_1 (but using nn.Sequential)
model_2 = nn.Sequential(
    nn.Linear(in_features=1, out_features=10),
    nn.Linear(in_features=10, out_features=10),
    nn.Linear(in_features=10, out_features=1)
).to(device)
#loss and optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model_2.parameters(), lr=0.1)

# Train the model
torch.manual_seed(42)

# Set the number of epochs
epochs = 1000

# Put data to target device
X_train_regression, y_train_regression = X_train_regression.to(device), y_train_regression.to(device)
X_test_regression, y_test_regression = X_test_regression.to(device), y_test_regression.to(device)

for epoch in range(epochs):
    ### Training
    # 1. Forward pass
    y_pred = model_2(X_train_regression)

    # 2. Calculate loss (no accuracy since it's a regression problem, not classification)
    loss = loss_fn(y_pred, y_train_regression)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_2.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_pred = model_2(X_test_regression)
        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test_regression)

    # Print out what's happening
    # if epoch % 100 == 0:
    #     print(f"Epoch: {epoch} | Train loss: {loss:.5f}, Test loss: {test_loss:.5f}")


#turn on evalution mode
model_2.eval()

#make predictions
with torch.inference_mode():
    y_preds = model_2(X_test_regression)

# plot_predictions(train_data=X_train_regression,
#                  train_labels=y_train_regression,
#                  test_data=X_test_regression,
#                  test_labels=y_test_regression,
#                  predictions=y_preds)
#
# plt.show()

# But how about we give it the capacity to draw non-straight (non-linear) lines?
# 6.1 Recreating non-linear data
#make and plot data
n_samples = 1000
X,y = make_circles(n_samples=1000, noise=0.03, random_state=42)
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu)
# plt.show()
#
# let's split it into training and test sets using 80% of the data for training and 20% for testing.
#turn data into tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

#split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6.2 Building a model with non-linearity
#build model with non linear activation func
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

model_3 = CircleModelV2().to(device)
# print(model_3)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_3.parameters(), lr=0.1)
# 6.3 Training a model with non-linearity
# Fit the model
torch.manual_seed(42)
epochs = 1000

# Put all data on target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    # 1. Forward pass
    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))  # logits -> prediction probabilities -> prediction labels

    # 2. Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train)  # BCEWithLogitsLoss calculates loss using logits
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_3.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_3(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))  # logits -> prediction probabilities -> prediction labels
        # 2. Calcuate loss and accuracy
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening
    # if epoch % 100 == 0:
    #     print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")

# 6.4 Evaluating a model trained with non-linear activation functions
#make predictions
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()

# Plot decision boundaries for training and test sets
# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.title('Train')
# plot_decision_boundary(model_1, X_train, y_train)# model_1 = no non-linearity
# plt.subplot(1,2,2)
# plt.title('Test')
# plot_decision_boundary(model_3, X_test, y_test) # model_3 = has non-linearity
# plt.show()


# 7. Replicating non-linear activation functions
# Let's start by creating a small amount of data.
#create a toy tensor (similar to the data going into out model(s))
A = torch.arange(-10, 10, 1, dtype=torch.float32)
# plt.plot(A)
# plt.show()

# Now let's see how the ReLU activation function influences it.
# And instead of using PyTorch's ReLU (torch.nn.ReLU), we'll recreate it ourselves.
# The ReLU function turns all negatives to 0 and leaves the positive values as they are.

#create ReLU func by hand
def relu(x):
    return torch.maximum(torch.tensor(0),x)

# plt.plot(relu(A))
# plt.show()
#lets build a func to replicate the sigmoid function with Pytorch

#create a custom sigmoid func
def sigmoid(x):
    return 1 / (1+torch.exp(-x))

# print(sigmoid(A))
# plt.plot(sigmoid(A))
# plt.show()

# 8. Putting things together by building a multi-class PyTorch model
# To begin a multi-class classification problem, let's create some multi-class data.
# To do so, we can leverage Scikit-Learn's make_blobs() method.
# This method will create however many classes (using the centers parameter) we want.
# Specifically, let's do the following:
# Create some multi-class data with make_blobs().
# Turn the data into tensors (the default of make_blobs() is to use NumPy arrays).
# Split the data into training and test sets using train_test_split().
# Visualize the data.

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

#set hyperparameters for data creation
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

#create multi class data
X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,#X features
                            centers=NUM_CLASSES,#y labels
                            cluster_std=1.5, # give the clusters a little shake up (try changing this to 1.0, the default)
                            random_state=RANDOM_SEED)

#Turn data into tensors
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
# print(X_blob[:5], y_blob[:5])

# split into train and test sets
X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob, test_size=0.2, random_state=RANDOM_SEED)

#plot data

# plt.figure(figsize=(10,7))
# plt.scatter(X_blob[:, 0], X_blob[:, 1],c=y_blob, cmap=plt.cm.RdYlBu)
# plt.show()

# 8.2 Building a multi-class classification model in PyTorch
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self,x):
        return self.linear_layer_stack(x)

model_4 = BlobModel(input_features=NUM_FEATURES,
                    output_features=NUM_CLASSES,
                    hidden_units=8).to(device)

# print(model_4)
# 8.3 Creating a loss function and optimizer for a multi-class PyTorch model
# Since we're working on a multi-class classification problem, we'll use the nn.CrossEntropyLoss() method as our loss function.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_4.parameters(), lr=0.1)

# Perform a single forward pass on the data (we'll need to put it to the target device for it to work)
# print(model_4(X_blob_train.to(device))[:5])

# How many elements in a single prediction sample?
# print(model_4(X_blob_train.to(device))[0].shape, NUM_CLASSES)

# 8.5 Creating a training and testing loop for a multi-class PyTorch model
# The only difference is that we'll be adjusting the steps to turn the model outputs (logits) to prediction probabilities
# (using the softmax activation function) and then to prediction labels (by taking the argmax of the output of the softmax activation function).

#fit the model
torch.manual_seed(42)

#set the number of epochs
epochs = 100

# put the data into target device
X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

for epoch in range(epochs):
    model_4.train()
    #forward pass
    y_logits = model_4(X_blob_train)# model outputs raw logits
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # go from logits -> prediction probabilities -> prediction labels
    #calculate loss and accuracy
    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)
    #optimizer zero grad
    optimizer.zero_grad()
    #loss backwards()
    loss.backward()
    #optimizer step
    optimizer.step()
    #Testing
    model_4.eval()
    with torch.inference_mode():
        #forward pass
        test_logits = model_4(X_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        #calculate test loss and accuracy
        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test,
                               y_pred=test_pred)

    # if epoch % 10 == 0:
    #     print(f'Epoch {epoch}, Loss {loss:.5f}, Acc {acc:.2f}%, Test loss {test_loss:.5f}, Test acc {test_acc:.2f}%')

# 8.6 Making and evaluating predictions with a PyTorch multi-class model
# make predictions
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)

# print(y_logits[:10])

# Though to evaluate them, they'll have to be in the same form as our labels (y_blob_test) which are in integer form.
# Let's convert our model's prediction logits to prediction probabilities (using torch.softmax()) then to prediction labels (by taking the argmax()
# of each sample).
#turn predicted logits in prediction probabilities
y_pred_probs = torch.softmax(y_logits, dim=1)

#turn prediction probabilities into predicted labels
y_preds = y_pred_lables.argmax(dim=1)

# plt.figure(figsize=(12,6))
# plt.subplot(1, 2, 1)
# plt.title('Train')
# plot_decision_boundary(model_4, X_blob_train, y_blob_train)
# plt.subplot(1, 2, 2)
# plt.title('Test')
# plot_decision_boundary(model_4, X_blob_test, y_blob_test)
# plt.show()

# 9. More classification evaluation metrics
# Accuracy	Out of 100 predictions, how many does your model get correct? E.g. 95% accuracy means it gets 95/100 predictions correct.	torchmetrics.Accuracy() or sklearn.metrics.accuracy_score()
# Precision	Proportion of true positives over total number of samples. Higher precision leads to less false positives (model predicts 1 when it should've been 0).	torchmetrics.Precision() or sklearn.metrics.precision_score()
# Recall	Proportion of true positives over total number of true positives and false negatives (model predicts 0 when it should've been 1). Higher recall leads to less false negatives.	torchmetrics.Recall() or sklearn.metrics.recall_score()
# F1-score	Combines precision and recall into one metric. 1 is best, 0 is worst.	torchmetrics.F1Score() or sklearn.metrics.f1_score()
# Confusion matrix	Compares the predicted values with the true values in a tabular way, if 100% correct, all values in the matrix will be top left to bottom right (diagnol line).	torchmetrics.ConfusionMatrix or sklearn.metrics.plot_confusion_matrix()
# Classification report	Collection of some of the main classification metrics such as precision, recall and f1-score.	sklearn.metrics.classification_report()

# Let's try the torchmetrics.Accuracy metric out.
# Setup metric and make sure it's on the target device
from torchmetrics import Accuracy
torchmetrics_accuracy = Accuracy(task='multiclass', num_classes=4).to(device)

# Calculate accuracy
# torchmetrics_accuracy(y_preds, y_blob_test)

# print(y_preds.shape, y_blob_test.shape)








