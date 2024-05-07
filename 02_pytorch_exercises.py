#1. Make a binary classification dataset with Scikit-Learn's make_moons() function.
# For consistency, the dataset should have 1000 samples and a random_state=42.
# Turn the data into PyTorch tensors. Split the data into training and test sets using train_test_split with 80% training and 20% testing.
from sklearn.datasets import make_moons

n_samples = 1000

#create circles
X, y = make_moons(n_samples, noise=0.03, random_state=42)

#turn data into tensors
import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

#split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#2. Build a model by subclassing nn.Module that incorporates non-linear activation functions and is capable of fitting the data you created in 1.
# Feel free to use any combination of PyTorch layers (linear and non-linear) you want.
import torch.nn as nn

#make device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.act = nn.ReLU()
        self.layer3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        x = self.act(x)
        x = self.layer3(x)
        return x

model_0 = CircleModel().to(device)

#3. Setup a binary classification compatible loss function and optimizer to use when training the model.
loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

#4. Create a training and testing loop to fit the model you created in 2 to the data you created in 1.
# To measure model accuracy, you can create your own accuracy function or use the accuracy function in TorchMetrics.
# Train the model for long enough for it to reach over 96% accuracy.
# The training loop should output progress every 10 epochs of the model's training and test set loss and accuracy.

#calculate accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred,).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

torch.manual_seed(42)

epochs = 1000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_0.train()
    #forward pass
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    #calculate loss/accuracy
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
    #     print(f'Epoch {epoch}, loss {loss:.5f}, Accuracy {acc:.2f}%, test loss {test_loss:.5f}, test acc {test_acc:.2f}%')


# Make predictions with your trained model and plot them using the plot_decision_boundary() function created in this notebook.
import requests
from pathlib import Path
import matplotlib.pyplot as plt

# Download helper functions from Learn PyTorch repo
if Path('helper_functions.py').is_file():
    print("helper_functions.py already exists, skipping download")
else:
    print("Downloading helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

# plot decision boundaries for training and testing sets
# plt.figure(figsize=(12,6))
# plt.subplot(1, 2, 1)
# plt.title('Train')
# plot_decision_boundary(model_0, X_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title('Test')
# plot_decision_boundary(model_0, X_test, y_test)
# plt.show()

#6. Replicate the Tanh (hyperbolic tangent) activation function in pure PyTorch
import numpy as np

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

#7. Create a multi-class dataset using the spirals data creation function from CS231n (see below for the code).
# Construct a model capable of fitting the data (you may need a combination of linear and non-linear layers).
# Build a loss function and optimizer capable of handling multi-class data (optional extension: use the Adam optimizer instead of SGD, you may have to experiment with different values of the learning rate to get it working).
# Make a training and testing loop for the multi-class data and train a model on it to reach over 95% testing accuracy (you can use any accuracy measuring function here that you like).
# Plot the decision boundaries on the spirals dataset from your model predictions, the plot_decision_boundary() function should work for this dataset too.

#Code for creating a spiral dataset from CS231n
N = 100 # number points per class
D = 2 #Dimensionality
K = 3 #number of classes
X = np.zeros((N*K,D)) #data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') #class labels
for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N)#radius
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2#theta
    X[ix] = np.c_[r * np.sin(t), r*np.cos(t)]
    y[ix] = j

# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()
















