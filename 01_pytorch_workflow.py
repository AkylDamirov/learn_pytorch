#pytorch workflow fundametals

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#1. Data (preparing and loading)
# Let's create our data as a straight line.
# We'll use linear regression to create the data with known parameters (things that can be learned by a model)' \
# and then we'll use PyTorch to see if we can build model to estimate these parameters using gradient descent.
#create know parameters
weight = 0.7
bias = 0.3

#create data
X = torch.arange(0, 1, 0.02).unsqueeze(dim=1).unsqueeze(dim=1)
y = weight * X + bias
# print(X[:10], y[:10])

#2 Split data into training and test sets

# We've got some data.
# But before we build a model we need to split it up.
# One of most important steps in a machine learning project is creating a training and test set (and when required, a validation set).
# Each split of the dataset serves a specific purpose:

# Training set	The model learns from this data (like the course materials you study during the semester).	~60-80%	Always
# Validation set	The model gets tuned on this data (like the practice exam you take before the final exam).	~10-20%	Often but not always
# Testing set	The model gets evaluated on this data to test what it has learned (like the final exam you take at the end of the semester).	~10-20%	Always
#
#create train/test split
train_split = int(0.8 * len(X)) #80% of data used for training set, 20% for testing
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
# print(len(X_train), len(y_train), len(X_test), len(y_test))

# Wonderful, we've got 40 samples for training (X_train & y_train) and 10 samples for testing (X_test & y_test).
# The model we create is going to try and learn the relationship between X_train & y_train and then we will evaluate what it learns on X_test and y_test.
# But right now our data is just numbers on a page.
# Let's create a function to visualize it.

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})
    plt.show()

# plot_predictions()

#BUILD MODEL
#Create a linear regression model class
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(1, dtype=torch.float), requires_grad=True)#start with random weights (this will get adjusted as the model learns)
        self.bias = nn.Parameter(torch.rand(1, dtype=torch.float), requires_grad=True)

    #forward defines the computation in model
    def forward(self, x): #x is input data
        return self.weights * x + self.bias #this is linear regression formula (y = m*x+b)

#PyTorch model building essentials
# PyTorch has four (give or take) essential modules you can use to create almost any kind of neural network you can imagine.
# They are torch.nn, torch.optim, torch.utils.data.Dataset and torch.utils.data.DataLoader.

# torch.nn	Contains all of the building blocks for computational graphs (essentially a series of computations executed in a particular way).
# torch.nn.Parameter	Stores tensors that can be used with nn.Module. If requires_grad=True gradients (used for updating model parameters via gradient descent) are calculated automatically, this is often referred to as "autograd".
# torch.nn.Module	The base class for all neural network modules, all the building blocks for neural networks are subclasses. If you're building a neural network in PyTorch, your models should subclass nn.Module. Requires a forward() method be implemented.
# torch.optim	Contains various optimization algorithms (these tell the model parameters stored in nn.Parameter how to best change to improve gradient descent and in turn reduce the loss).
# def forward()	All nn.Module subclasses require a forward() method, this defines the computation that will take place on the data passed to the particular nn.Module (e.g. the linear regression formula above).

# Checking the contents of a PyTorch model
#set manual seed since nn.Parameter are randomly initialized
torch.manual_seed(41)

#create an instance of the model (this is subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel()

#Check the nn.Parameter(s) within the nn.Module subclass we created
# print(list(model_0.parameters()))

# We can also get the state (what the model contains) of the model using .state_dict().
#list named parameters
# print(model_0.state_dict())

# Making predictions using torch.inference_mode()
#make predictions with model
with torch.inference_mode():
    y_preds = model_0(X_test)

# Check the predictions
# print(f"Number of testing samples: {len(X_test)}")
# print(f"Number of predictions made: {len(y_preds)}")
# print(f"Predicted values:\n{y_preds}")

# Our predictions are still numbers on a page, let's visualize them with our plot_predictions() function we created above.
# plot_predictions(predictions=y_preds)








