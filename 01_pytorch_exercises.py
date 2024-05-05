import torch
import torch.nn as nn
import matplotlib.pyplot as plt

#1. Create a straight line dataset using the linear regression formula (weight * X + bias).
# Set weight=0.3 and bias=0.9 there should be at least 100 datapoints total.
# Split the data into 80% training, 20% testing.
# Plot the training and testing data so it becomes visual.
weight = 0.3
bias = 0.9

X = torch.arange(0, 1, 0.02).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training data')
    plt.scatter(test_data, test_labels, c='g', s=4, label='Testing data')

    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={'size':14})
    plt.show()

# plot_predictions()

#2. Build a PyTorch model by subclassing nn.Module.
# Inside should be a randomly initialized nn.Parameter() with requires_grad=True, one for weights and one for bias.
# Implement the forward() method to compute the linear regression function you used to create the dataset in 1.
# Once you've constructed the model, make an instance of it and check its state_dict().
# Note: If you'd like to use nn.Linear() instead of nn.Parameter() you can.

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.rand(1, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        return self.weights * x + self.bias

torch.manual_seed(42)
model_1 = LinearRegressionModel()
# model_state = model_1.state_dict()
# print(model_state)

#3. Create a loss function and optimizer using nn.L1Loss() and torch.optim.SGD(params, lr) respectively.
# Set the learning rate of the optimizer to be 0.01 and the parameters to optimize should be the model parameters from the model you created in 2.
# Write a training loop to perform the appropriate training steps for 300 epochs.
# The training loop should test the model on the test dataset every 20 epochs.

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

torch.manual_seed(42)

epochs = 300


train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    model_1.train()

    y_pred = model_1(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss_values.append(loss.item())

    if epoch % 20 == 0:
        epoch_count.append(epoch)
        model_1.eval()
        with torch.no_grad():
            test_pred = model_1(X_test)
            test_loss = loss_fn(test_pred, y_test)
            test_loss_values.append(loss.item())

        # print(f'Epoch {epoch}, Train loss: {loss.item()}, Test loss {test_loss.item()}')

#
#4. Make predictions with the trained model on the test data.
# Visualize these predictions against the original training and testing data (note: you may need to make sure the predictions are not on the GPU
# if you want to use non-CUDA-enabled libraries such as matplotlib to plot).
model_1.eval()

with torch.inference_mode():
    y_preds = model_1(X_test)

# plot_predictions(predictions=y_preds)


#5. Save your trained model's state_dict() to file.
# Create a new instance of your model class you made in 2. and load in the state_dict() you just saved to it.
# Perform predictions on your test data with the loaded model and confirm they match the original model predictions from 4.

from pathlib import Path

MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = '01_pytorch_workflow_model_0_exercise.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)

loaded_model_1 = LinearRegressionModel()
loaded_model_1.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

loaded_model_1.eval()

with torch.inference_mode():
    loaded_model_preds = loaded_model_1(X_test)
#
# print(y_preds==loaded_model_preds)
# print(y_preds)
# print(loaded_model_preds)








