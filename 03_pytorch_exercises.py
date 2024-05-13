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
train_time_start_model_2 = timer()
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f'Epoch: {epoch}\n----')
    train_step(data_loader=train_dataloader,
               model=model_2,
               loss_fn=loss_fn,
               optimizer=optimizer,
               accuracy_fn=accuracy_fn,
               device=device
               )
    test_step(data_loader=test_dataloader,
              model=model_2,
              loss_fn=loss_fn,
              accuracy_fn=accuracy_fn,
              device=device)

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_2, end=train_time_end_model_2, device=device)


#10. Make predictions using your trained model and visualize at least 5 of them comparing the prediciton to the target label.

def make_predictions(model, data, device: torch.device=device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            #prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device)

            #forward pass
            pred_logit = model(sample)

            #get prediction probability
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            # Get pred_prob off GPU for further calculations
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

# Make predictions on test samples with model 2
pred_probs = make_predictions(model=model_2, data=test_samples)

# Turn the prediction probabilities into prediction labels by taking the argmax()
pred_classes = pred_probs.argmax(dim=1)

#plot predictions
# plt.figure(figsize=(9,9))
# nrows = 3
# ncols = 3
# for i, sample in enumerate(test_samples):
#     #create subplot
#     plt.subplot(nrows, ncols, i+1)
#
#     #plot the target image
#     plt.imshow(sample.squeeze(), cmap='gray')
#
#     # Find the prediction label
#     pred_label = class_names[pred_classes[i]]
#
#     # Get the truth label
#     truth_label = class_names[test_labels[i]]
#
#     # Create the title text of the plot
#     title_text = f'Pred {pred_label} Truth {truth_label}'
#
#     # Check for equality and change title colour accordingly
#     if pred_label==truth_label:
#         plt.title(title_text, fontsize=10, c='g')
#     else:
#         plt.title(title_text, fontsize=10, c='r')
#     plt.axis(False)
# plt.show()

#11. Plot a confusion matrix comparing your model's predictions to the truth labels.
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
        X,y = X.to(device), y.to(device)

        #forward pass
        y_logit = model_2(X)

        # Turn predictions from logits -> prediction probabilities -> predictions labels
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

        # Put predictions on CPU for evaluation
        y_preds.append(y_pred.cpu())

# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)

# Make a confusion matrix using torchmetrics.ConfusionMatrix.
import torchmetrics, mlxtend
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# 2. Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor, target=test_data.targets)

#plot confusion matrix
# fig, ax = plot_confusion_matrix(
#     conf_mat=confmat_tensor.numpy(),
#     class_names=class_names,
#     figsize=(10, 7)
# )
# plt.show()

#12. Create a random tensor of shape [1, 3, 64, 64] and pass it through a nn.Conv2d() layer with various
# hyperparameter settings (these can be any settings you choose), what do you notice if the kernel_size parameter goes up and down?
# torch.manual_seed(42)
# images_0 = torch.rand(size=(1,3,64,64))
# test_image = images_0[0] # get a single image for testing
#
# conv_layer = nn.Conv2d(in_channels=3,
#                        out_channels=10,
#                        kernel_size=3,
#                        stride=1,
#                        padding=0)
#
# print(conv_layer(test_image))
#13. Use a model similar to the trained model_2 from this notebook to make predictions on the test torchvision.datasets.FashionMNIST dataset.
# Then plot some predictions where the model was wrong alongside what the label of the image should've been.
# After visualing these predictions do you think it's more of a modelling error or a data error?
# As in, could the model do better or are the labels of the data too close to each other (e.g. a "Shirt" label is too close to "T-shirt/top")?


#load FashionMNIST dataset
fashion_mnist_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

fashion_mnist_test = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

#create dataloaders
fashion_train_dataloader = DataLoader(fashion_mnist_train, batch_size=BATCH_SIZE, shuffle=True)
fashion_test_dataloader = DataLoader(fashion_mnist_test, batch_size=BATCH_SIZE, shuffle=False)

#create a model similar to model_2
#its above (fashionMNISTmodelV2)

#initialize the fashionMNIST model
fashion_model = FashionMNISTModelV2(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)

#make predictions on the test set
fashion_model.eval()
fashion_preds = []
fashion_labels = []
with torch.no_grad():
    for images, labels in tqdm(fashion_test_dataloader, desc='Making predictions'):
        images, labels = images.to(device), labels.to(device)
        outputs = fashion_model(images)
        preds = torch.argmax(outputs, dim=1)
        fashion_preds.extend(preds.cpu())
        fashion_labels.extend(labels.cpu())

#convert prediction and labels to numpy arrays
fashion_preds = torch.tensor(fashion_preds).numpy()
fashion_labels = torch.tensor(fashion_labels).numpy()

# Plot some predictions where the model was wrong alongside the correct labels
# plt.figure(figsize=(12, 8))
#
# for i in range(9):
#     plt.subplot(3, 3, i+1)
#     plt.imshow(fashion_mnist_test[i][0].squeeze(), cmap='gray')
#     plt.title(f"Predicted: {class_names[fashion_preds[i]]}, Actual: {class_names[fashion_labels[i]]}")
#     plt.axis('off')
# plt.show()










