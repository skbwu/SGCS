# standard data science
import numpy as np
import pandas as pd

# standard pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# PyTorch data utilities
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
import utils
from utils import MNIST_MLP

# logging + I/O
import sys, copy, os, shutil, time
from importlib import reload


# reload our scripts
reload(utils)

# how many epochs are we training for? no more than 50. also what's our batch size?
epochs, batch_size = 100, 256

# command-line arguments
dataset = ["MNIST", "FashionMNIST"][int(sys.argv[1])]
num_layers = int(sys.argv[2])
seed = int(sys.argv[3])

# load our data
trainloader, testloader, data_dim = utils.load_data(dataset, batch_size)

# set a seed, instantiate our model + define loss function, optimizer
torch.manual_seed(seed)
model = utils.MNIST_MLP(num_layers=num_layers, data_dim=data_dim)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# create a directory in our models folder for this run
if dataset not in os.listdir("models"):
    os.mkdir(f"models/{dataset}")
model_name = f"mlp_num-layers={num_layers}_seed={str(seed).zfill(3)}"
if model_name not in os.listdir(f"models/{dataset}"):
    os.mkdir(f"models/{dataset}/{model_name}")

    
###### OUR TRAINING PIPELINE

# metrics to record for each epoch
train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []

# iterate through our epochs
for epoch in range(epochs):
    
    # initialize the RUNNING TRAINING LOSS list and TRAIN accuracy list
    running_loss, accuracy = [], [0, 0] # [# correct, total # of samples seen]
    
    # iterate through training data: train + collect train loss
    for data in trainloader:
        
        # reset gradient + get our data for this batch
        optimizer.zero_grad()
        inputs, labels = data
        
        # forward prop, backward prop, make incremental step
        outputs = model(inputs); loss = loss_func(outputs, labels)
        loss.backward(); optimizer.step()

        # update our running_loss (TRAINING!)
        running_loss.append(loss.item())
        
         # calculate + record our train acuracy (TRAINING!)
        with torch.no_grad():

            # get our predictions with the current weights + count no. of correct
            _, predicted = torch.max(outputs.data, 1)
            accuracy[1] += labels.size(0)
            accuracy[0] += (predicted == labels).sum().item()
    
    # compute mean train loss across batches, also test accuracy + mean test loss across batches
    with torch.no_grad():
                
        # add our training loss + accuracy to our lists
        train_losses.append(np.mean(np.array(running_loss)))
        train_accuracies.append((accuracy[0] / accuracy[1]))

        # initialize the RUNNING TEST LOSS + test accuracy list
        running_test_loss, test_accuracy = [], [0, 0]

        # compute test set metrics
        for test_data in testloader:

            # make test predictions + record running loss
            test_images, test_labels = test_data
            test_outputs = model(test_images)
            test_loss = loss_func(test_outputs, test_labels)
            running_test_loss.append(float(test_loss))
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_accuracy[1] += test_labels.size(0)
            test_accuracy[0] += (test_predicted == test_labels).sum().item()

        # add our test loss/accuracies to our lists
        test_losses.append(np.mean(np.array(running_test_loss)))
        test_accuracies.append((test_accuracy[0] / test_accuracy[1]))
        
        # save our weights at every epoch!
        torch.save(obj=model.state_dict(), f=f"models/{dataset}/{model_name}/{str(epoch).zfill(3)}.pth")
        
# at the very end, save our logs for this model
logs = pd.DataFrame(data=np.array([list(np.arange(len(train_losses))), train_losses, 
                                   test_losses, train_accuracies, test_accuracies]).T,
                    columns=["epoch", "train_loss", "test_loss", "train_acc", "test_acc"])
logs.to_csv(f"models/{dataset}/{model_name}/logs.csv", index=False)