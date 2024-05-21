# Recreate the data loading functions we built in sections 1, 2, 3 and 4. You should have train and test DataLoader's ready to use.
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Get data
import requests
import zipfile
from pathlib import Path

#old
# #setup path to data folder
# data_path = Path('data/')
# image_path = data_path / 'pizza_steak_sushi'
#
# if image_path.is_dir():
#     print(f'{image_path} is exist')
# else:
#     print(f'Did not find {image_path}, creating one...')
#     image_path.mkdir(parents=True, exist_ok=True)
#
#     #download
#     with open(data_path / "pizza_steak_sushi.zip", 'wb') as f:
#         request = requests.get('https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip')
#         print('Downloading pizza steak sushi data...')
#         f.write(request.content)
#
#     #unzip
#     with zipfile.ZipFile(data_path / 'pizza_steak_sushi.zip', 'r') as zip_ref:
#         print('Unzipping data')
#         zip_ref.extractall(image_path)

#new one
#setup path to data folder
data_path = Path('data/')
image_path = data_path / 'pizza_steak_sushi_20'

if image_path.is_dir():
    print(f'{image_path} is exist')
else:
    print(f'Did not find {image_path}, creating one...')
    image_path.mkdir(parents=True, exist_ok=True)

    #download
    with open(data_path / "pizza_steak_sushi_20.zip", 'wb') as f:
        request = requests.get('https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip')
        print('Downloading pizza steak sushi data...')
        f.write(request.content)

    #unzip
    with zipfile.ZipFile(data_path / 'pizza_steak_sushi_20.zip', 'r') as zip_ref:
        print('Unzipping data')
        zip_ref.extractall(image_path)

# 2. data preparation
#setup train and testing path
train_dir = image_path / 'train'
test_dir = image_path / 'test'

# 3. Transforming data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#write transform for image
data_transform = transforms.Compose([
    #Resize image to 64x64
    transforms.Resize((64, 64)),
    #flip images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    #turn image into torch.tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])

# 4. Option 1: Loading Image Data Using ImageFolder
#use ImageFolder to create dataset(s)

from torchvision import datasets

train_data = datasets.ImageFolder(root=train_dir,# target folder of images
                                  transform=data_transform,# transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

# print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

#get classes names as a list
class_names = train_data.classes

#get class names as a dict
class_dict = train_data.class_to_idx

img, label = train_data[0][0], train_data[0][1]

# 4.1 Turn loaded images into DataLoader's
#turn train and test Datasets into Dataloaders
from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=1,
                              num_workers=0,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=1,
                             num_workers=0,
                             shuffle=False)

img,label = next(iter(train_dataloader))


# 7. Model 0: TinyVGG without data augmentation

simple_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

#load and transform data
from torchvision import datasets
train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

#turn data into dataloaders
import os

#setup batch size and number of workers
BATCH_SIZE = 32
# NUM_WORKERS = os.cpu_count()

#create dataloaders
train_dataloader_simple = DataLoader(train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=0)

test_dataloader_simple = DataLoader(test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=0)

#3. Recreate model_0 we built in section 7.
# 7.2 Create TinyVGG model class
class TinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
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
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16, out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_data.classes)).to(device)

#4. Create training and testing functions for model_0.

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn,
               optimizer):
    #put model in train mode
    model.train()

    #setup train loss and train accuracy
    train_loss, train_acc = 0,0

    #loop through data loader data batches
    for batch, (X,y) in enumerate(dataloader):
        #send data to target device
        X,y = X.to(device), y.to(device)

        #forward pass
        y_pred = model(X)

        #calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        #optimizer zero grad
        optimizer.zero_grad()

        #loss backward
        loss.backward()

        #optimizer step
        optimizer.step()

        #calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    #adjust metrics to get average loss and acc per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    #put model on eval mode
    model.eval()

    #setup loss and accuracy
    test_loss, test_acc = 0,0

    with torch.inference_mode():
        #loop through Dataloaders batches
        for batch, (X,y) in enumerate(dataloader):
            #send data to target device
            X,y = X.to(device), y.to(device)

            #forward pass
            test_pred_logits = model(X)

            #calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            #calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

        #adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc

# 7.6 Creating a train() function to combine train_step() and test_step()
from tqdm.auto import tqdm

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    results = {'train_loss':[],
               'train_acc':[],
               'test_loss':[],
               'test_acc':[]}

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

    return results

#5. Try training the model you made in exercise 3 for 5, 20 and 50 epochs, what happens to the results?
# Use torch.optim.Adam() with a learning rate of 0.001 as the optimizer.

#set random seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

#set number of epochs
NUM_EPOCHS = 20

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

# train model 0
model_0_results = train(model=model_0,
                        train_dataloader=train_dataloader_simple,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)



# Double the number of hidden units in your model and train it for 20 epochs, what happens to the results?
# with 10
# Epoch: 20 | train_loss: 0.7857 | train_acc: 0.5703 | test_loss: 1.0661 | test_acc: 0.4848

# with 20 hidden units
# Epoch: 20 | train_loss: 0.6956 | train_acc: 0.7461 | test_loss: 1.1708 | test_acc: 0.4148

# 7. Double the data you're using with your model and train it for 20 epochs, what happens to the results?
# Note: You can use the custom data creation notebook to scale up your Food101 dataset.
# You can also find the already formatted double data (20% instead of 10% subset) dataset on GitHub,
# you will need to write download code like in exercise 2 to get it into this notebook.

# with 20% data
# Epoch: 20 | train_loss: 0.6821 | train_acc: 0.7125 | test_loss: 0.8991 | test_acc: 0.5557

#8. Make a prediction on your own custom image of pizza/steak/sushi (you could even download one from the internet) and share your prediction.
# Does the model you trained in exercise 7 get it right?
# If not, what do you think you could do to improve it?
image_address = 'https://imgproxy.sbermarket.ru/imgproxy/size-680-680/czM6Ly9jb250ZW50LWltYWdlcy1wcm9kL3Byb2R1Y3RzLzMxNTkzMjg3L29yaWdpbmFsLzEvMjAyNC0wMi0xNlQyMyUzQTE1JTNBMzEuMzM1NTg3JTJCMDAlM0EwMC8zMTU5MzI4N18xLmpwZw==.jpg?raw=true'
custom_image_path = data_path / 'pizza_2.jpeg'


if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # When downloading from GitHub, need to use the "raw" file link
        request = requests.get(image_address) ### <- CHANGED
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

# Since we want to load in an image, we'll use torchvision.io.read_image().
import torchvision
import matplotlib.pyplot as plt
def pred_and_plot_image(model: torch.nn.Module,
                        image_path,
                        class_names,
                        transform=None,
                        device: torch.device=device):
    # Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # Divide the image pixel values by 255 to get them between[0, 1]
    target_image = target_image / 255

    #transform in necessary
    if transform:
        target_image = transform(target_image)

    #make sure if model in on the target device
    model.to(device)

    model.eval()
    with torch.inference_mode():
        #add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # Convert logits -> prediction probabilities(using torch.softmax() for multi -class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1,2,0))
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show()

custom_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
])

#pred on out custom image
pred_and_plot_image(model=model_0,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=custom_image_transform,
                    device=device)