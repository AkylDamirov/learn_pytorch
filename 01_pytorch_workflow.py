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

#Train model
#creating a loss func and optimizer
# Loss function	- Measures how wrong your models predictions (e.g. y_preds) are compared to the truth labels (e.g. y_test).
# Lower the better.	PyTorch has plenty of built-in loss functions in torch.nn.	Mean absolute error (MAE) for regression problems (torch.nn.L1Loss()).
# Binary cross entropy for binary classification problems (torch.nn.BCELoss()).

# Optimizer	- Tells your model how to update its internal parameters to best lower the loss.
# You can find various optimization function implementations in torch.optim.
# Stochastic gradient descent (torch.optim.SGD()). Adam optimizer (torch.optim.Adam()).

#create a loss function
loss_fn = nn.L1Loss() #MAE loss is same as L1Loss()

#create an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(),#parameters of target model to optimize
                            lr=0.01)#learning rate (how much optimizer should change parameters at each step, higher=more (less stable),lower=less (might take a long time))

#Training loop
# For the training loop, we'll build the following steps:
#  1	Forward pass	The model goes through all of the training data once, performing its forward() function calculations.	model(x_train)
#  2	Calculate the loss	The model's outputs (predictions) are compared to the ground truth and evaluated to see how wrong they are.	loss = loss_fn(y_pred, y_train)
#  3	Zero gradients	The optimizers gradients are set to zero (they are accumulated by default) so they can be recalculated for the specific training step.	optimizer.zero_grad()
#  4	Perform backpropagation on the loss	Computes the gradient of the loss with respect for every model parameter to be updated (each parameter with requires_grad=True). This is known as backpropagation, hence "backwards".	loss.backward()
#  5	Update the optimizer (gradient descent)	Update the parameters with requires_grad=True with respect to the loss gradients in order to improve them.	optimizer.step()

#Testing loop
# As for the testing loop (evaluating our model), the typical steps include:
# 1	Forward pass	The model goes through all of the training data once, performing its forward() function calculations.	model(x_test)
# 2	Calculate the loss	The model's outputs (predictions) are compared to the ground truth and evaluated to see how wrong they are.	loss = loss_fn(y_pred, y_test)
# 3	Calulate evaluation metrics (optional)	Alongisde the loss value you may want to calculate other evaluation metrics such as accuracy on the test set.	Custom functions

torch.manual_seed(42)
#number of epichs
epochs = 100

#create empty lost lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

#pass the data through the model for a number of epochs (e.g. 100)
for epoch in range(epochs):
    #Training
    #put in training mode
    model_0.train()

    #1 forward pass on train data using the forward() method inside
    y_pred = model_0(X_train)

    #2 calculate the loss
    loss = loss_fn(y_pred, y_train)

    #3 zero grad of the optimizer
    optimizer.zero_grad()

    #4  Loss backwards()
    loss.backward()

    #5 progress the optimizer
    optimizer.step()

    #Testing
    #put the model on evaluation model
    model_0.eval()

    with torch.inference_mode():
        #1 forward pass on test data
        test_pred = model_0(X_test)

        #2 calculate loss on test data
        test_loss = loss_fn(test_pred, y_test.type(torch.float)) # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

        # #print out whats happening
        # if epoch % 10 == 0:
        #     epoch_count.append(epoch)
        #     train_loss_values.append(loss.detach().numpy())
        #     test_loss_values.append(test_loss.detach().numpy())
        #     print(f'Epoch {epoch} | MAE train loss {loss} | MAE test loss {test_loss}')

# #plot the loss curves
# plt.plot(epoch_count, train_loss_values, label='Train loss')
# plt.plot(epoch_count, test_loss_values, label='Test loss')
# plt.title('Train and Test loss curves')
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.legend()
# plt.show()


# Making predictions with a trained PyTorch model (inference)
# There are three things to remember when making predictions (also called performing inference) with a PyTorch model:
#
# Set the model in evaluation mode (model.eval()).
# Make the predictions using the inference mode context manager (with torch.inference_mode(): ...).
# All predictions should be made with objects on the same device (e.g. data and model on GPU only or data and model on CPU only).

#set the model in evaluation mode
model_0.eval()

#setup the inference mode context manager
with torch.inference_mode():
    y_preds = model_0(X_test)

# print(y_preds)
# plot_predictions(predictions=y_preds)

 # Saving and loading a PyTorch model
# three main methods:
# torch.save	Saves a serialized object to disk using Python's pickle utility. Models, tensors and various other Python objects like dictionaries can be saved using torch.save.
# torch.load	Uses pickle's unpickling features to deserialize and load pickled Python object files (like models, tensors or dictionaries) into memory. You can also set which device to load the object to (CPU, GPU etc).
# torch.nn.Module.load_state_dict	Loads a model's parameter dictionary (model.state_dict()) using a saved state_dict() object.

# Saving a PyTorch model's state_dict()
# Let's see how we can do that in a few steps:
#
# We'll create a directory for saving models to called models using Python's pathlib module.
# We'll create a file path to save the model to.
# We'll call torch.save(obj, f) where obj is the target model's state_dict() and f is the filename of where to save the model.

from pathlib import Path

#1 create models directory
MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2 crate model save path
MODEL_NAME = '01_pytorch_workflow_model_0.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# # 3 save the model state dict
# print(f'saving model to {MODEL_SAVE_PATH}')
# torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
#            f=MODEL_SAVE_PATH)


# Loading a saved PyTorch model's state_dict()
# Since we've now got a saved model state_dict() at models/01_pytorch_workflow_model_0.pth we can now load it in using torch.nn.Module.load_state_dict(torch.load(f))
# where f is the filepath of our saved model state_dict().
# Why call torch.load() inside torch.nn.Module.load_state_dict()?
# Because we only saved the model's state_dict() which is a dictionary of learned parameters and not the entire model,
# we first have to load the state_dict() with torch.load() and then pass that state_dict() to a new instance of our model (which is a subclass of nn.Module).

# The disadvantage of this approach (saving the whole model) is that the serialized data is bound to the specific classes and
# the exact directory structure used when the model is saved...
# Because of this, your code can break in various ways when used in other projects or after refactors.

# Let's test it out by created another instance of LinearRegressionModel(),'
#which is a subclass of torch.nn.Module and will hence have the in-built method load_state_dict().

#instantiate a new instance of our model (this will instantiated with random weights)

loaded_model_0 = LinearRegressionModel()

#load the state_dict of your saved model (this will update the new instance of our model with trained weights)
loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Now to test our loaded model, let's perform inference with it (make predictions) on the test data.

# 1 put the loaded model into evaluation mode
loaded_model_0.eval()

# 2 use the inference mode context manager to make predictions
with torch.inference_mode():
    loaded_model_preds = loaded_model_0(X_test) # perform a forward pass on the test data with the loaded model

# Compare previous model predictions with loaded model predictions (these should be the same)
# print(y_preds==loaded_model_preds)


