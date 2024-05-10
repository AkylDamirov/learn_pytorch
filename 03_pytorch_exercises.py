#2 Search "what is overfitting in machine learning" and write down a sentence about what you find.
# overfitting its when model trained a lot with specific or small amount of data and gives accurate forecasts in train data but not for new (test data)

# 3. Search "ways to prevent overfitting in machine learning", write down 3 of the things you find and a sentence about each
#you can prevent overfitting diversifying and scaling set of data and use other strategies
#early stop
# Early Stop pauses the training phase before the machine learning model learns the
# noise in the data. However, it is important to choose the right time, otherwise the model will still not give accurate results.
# Trimming
# When building a model, you can define several objects or parameters that affect the final prediction. Feature sampling (or pruning)
# identifies the most important features in the training set and eliminates unnecessary ones. For example, to predict whether an image is an animal or a human, you can look at
# various input parameters such as face shape, ear position, body structure, etc. You can give preference to the face shape and ignore the eye shape.

# Regularization
# Regularization is a set of learning/optimization techniques aimed at reducing overfitting. These methods attempt to eliminate those factors that do not influence the prediction results by ranking objects
# based on importance. For example, mathematical calculations apply penalties to objects with minimal impact. Consider a statistical model that tries to predict house prices in a city 20 years from now. Regularization will give a smaller penalty value
# for characteristics such as population growth and average annual income, but a larger penalty value for the average annual temperature in a city.

# Ensemble
# An ensemble combines the predictions of several individual machine learning algorithms. Some models are called weak because their results are often inaccurate. Ensemble methods combine all weak learners to obtain more accurate results.
# They use multiple models to analyze sample data and select the most accurate results. The two main ensemble methods are batching and boosting. Boosting trains different machine learning models one by one to produce the final result, while batching trains them in parallel.
#
# Data Augmentation
# Data Augmentation is a machine learning technique in which sample data changes slightly each time the model processes it. This can be done by slightly changing the input data. With moderate data augmentation, training sets appear to be unique to the model and prevent the model
# from learning their characteristics. For example, applying transformations such as translation, reflection, and rotation to input images.


# 4.Spend 20-minutes reading and clicking through the CNN Explainer website.
# Upload your own example image using the "upload" button and see what happens in each layer of a CNN as your image passes through it.


#5. Load the torchvision.datasets.MNIST() train and test datasets.
import torch
import torch.nn as nn

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
#setup train data

train_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

# setup testing data
test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

#6. Visualize at least 5 different samples of the MNIST training dataset.
class_names = train_data.classes

# torch.manual_seed(42)
# fig = plt.figure(figsize=(9,9))
# rows, cols = 3,3
# for i in range(1, rows*cols+1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img,label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap='gray')
#     plt.title(class_names[label])
#     plt.axis(False)
#
# plt.show()

# 7. Turn the MNIST train and test datasets into dataloaders using torch.utils.data.DataLoader, set the batch_size=32.
from torch.utils.data import DataLoader

#setup the batch size hyperparameter
BATCH_SIZE = 32

#turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# Recreate model_2 used in this notebook (the same model from the CNN Explainer website, also known as TinyVGG) capable of fitting on the MNIST dataset.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
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
            nn.Linear(in_features=hidden_units*7*7,out_features=output_shape)
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

#9. Train the model you built in exercise 8. on CPU and GPU and see how long it takes on each.

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn,
               optimizer,
               accuracy_fn,
               device: torch.device=device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        X,y = X.to(device), y.to(device)
        #forward pass
        y_pred = model(X)
        #calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f'Train loss {train_loss:.5f} Train accuracy {train_acc:.2f}%')

def test_step(data_loader, model, loss_fn, accuracy_fn, device: torch.device=device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X,y = X.to(device), y.to(device)

            #forward pass
            test_pred = model(X)

            #calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        #adjust metrics and print
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f'Test loss {test_loss:.5f}, Test accuracy {test_acc:.2f}%')


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

torch.manual_seed(42)
#measure time
from timeit import default_timer as timer
from tqdm.auto import tqdm
from helper_functions import accuracy_fn

def print_train_time(start, end, device: torch.device=None):
    total_time = end-start
    print(f'Train time on {device} {total_time:.3f} seconds')
    return total_time



#train and test model
# train_time_start_model_2 = timer()
# epochs = 3
# for epoch in tqdm(range(epochs)):
#     print(f'Epoch: {epoch}\n----')
#     train_step(data_loader=train_dataloader,
#                model=model_2,
#                loss_fn=loss_fn,
#                optimizer=optimizer,
#                accuracy_fn=accuracy_fn,
#                device=device
#                )
#     test_step(data_loader=test_dataloader,
#               model=model_2,
#               loss_fn=loss_fn,
#               accuracy_fn=accuracy_fn,
#               device=device)
#
# train_time_end_model_2 = timer()
# total_train_time_model_2 = print_train_time(start=train_time_start_model_2, end=train_time_end_model_2, device=device)


#10. Make predictions using your trained model and visualize at least 5 of them comparing the prediciton to the target label.









