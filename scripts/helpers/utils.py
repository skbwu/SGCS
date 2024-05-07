# PyTorch data utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST  

# some standard data augmentation strategies
train_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                       transforms.RandomHorizontalFlip(), 
                                       transforms.ToTensor()])

# helper function for loading in our data with standard transforms
def load_data(dataset, batch_size):

    # which dataset are we running?
    if dataset == "MNIST":
        data_train = MNIST(root="./data", train=True, download=True, transform=train_transforms)
        data_test = MNIST(root="./data", train=False, download=True, transform=train_transforms)
        data_dim = 1024
    elif dataset == "FashionMNIST":
        data_train = FashionMNIST(root="./data", train=True, download=True, transform=train_transforms)
        data_test = FashionMNIST(root="./data", train=False, download=True, transform=train_transforms)
        data_dim = 1024
    elif dataset == "CIFAR10":
        data_train = CIFAR10(root="./data", train=True, download=True, transform=train_transforms)
        data_test = CIFAR10(root="./data", train=False, download=True, transform=train_transforms)
        data_dim = 3072

    # put our data into a dataloader for better memory management
    trainloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False)

    # return as output our trainloader and valloader, and data_dim
    return trainloader, valloader, data_dim


'''
1 hidden layers: 101770 parameters.
2 hidden layers: 118282 parameters.
3 hidden layers: 134794 parameters
'''
# create our MLP model template for MNIST and FashionMNIST
class MNIST_MLP(nn.Module):
    
    # constructor
    def __init__(self, num_layers, data_dim):
        
        # call parent constructor
        super().__init__()
        
        # instantiate our layers, depending on how many hidden layers we want
        if num_layers == 1:
            self.linear1 = nn.Linear(in_features=data_dim, out_features=128, bias=True)
            self.linear2 = nn.Linear(in_features=128, out_features=10, bias=True)
        elif num_layers == 2:
            self.linear1 = nn.Linear(in_features=data_dim, out_features=128, bias=True)
            self.linear2 = nn.Linear(in_features=128, out_features=128, bias=True)
            self.linear3 = nn.Linear(in_features=128, out_features=10, bias=True)
        elif num_layers == 3:
            self.linear1 = nn.Linear(in_features=data_dim, out_features=128, bias=True)
            self.linear2 = nn.Linear(in_features=128, out_features=128, bias=True)
            self.linear3 = nn.Linear(in_features=128, out_features=128, bias=True)
            self.linear4 = nn.Linear(in_features=128, out_features=10, bias=True)
            
        # encode the parameter
        self.num_layers = num_layers
            
    # create our forward function -- let's use relu
    def forward(self, x):
        
        # reshape our data first for linear architecture, but respect batch size
        x = x.view(x.size(0), -1)
        
        # do our forward pass based on how many layers we have
        if self.num_layers == 1:
            x = self.linear2(F.relu(self.linear1(x)))
        elif self.num_layers == 2:
            x = self.linear3(F.relu(self.linear2(F.relu(self.linear1(x)))))
        elif self.num_layers == 3:
            x = self.linear4(F.relu(self.linear3(F.relu(self.linear2(F.relu(self.linear1(x)))))))
            
        # return our output
        return x
    
    
'''
2 modules: 14110 parameters.
3 modules: 5130 parameters.
4 modules: 2750 parameters.
'''
class MNIST_CNN(nn.Module):
    def __init__(self, num_modules):
        super().__init__()
        
        # record our param
        self.num_modules = num_modules
        
        # first conv + pool module
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        # add second conv + pool module ONLY IF num_modules >= 2
        if num_modules >= 2:
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32, padding=1)
            self.conv3 = nn.Conv2d(in_channels=32, out_channels=20, kernel_size=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

         # add third conv + pool module ONLY IF num_modules >= 3
        if num_modules >= 3:
            self.conv4 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, groups=20, padding=1)
            self.conv5 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=1)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

         # add fourth conv + pool module ONLY IF num_modules >= 4
        if num_modules >= 4:
            self.conv6 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, groups=20, padding=1)
            self.conv7 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=1)
            self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)

        # final linear output layer
        if num_modules == 2:
            self.linear = nn.Linear(in_features=1280, out_features=10)
        elif num_modules == 3:
            self.linear = nn.Linear(in_features=320, out_features=10)
        elif num_modules == 4:
            self.linear = nn.Linear(in_features=20, out_features=10)

    # governs the forward-pass
    def forward(self, x):
        
        # always do this part
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)

        # checkpointing just like constructor
        if self.num_modules >= 2:
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.pool2(x)
            x = F.relu(x)
        if self.num_modules >= 3:
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.pool3(x)
            x = F.relu(x)
        if self.num_modules >= 4:
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.pool4(x)
            x = F.relu(x)
        
        # reshape based on the number of convolutional layers
        if self.num_modules == 2:
            x = x.reshape(-1, 1280)
        elif self.num_modules == 3:
            x = x.reshape(-1, 320)
        elif self.num_modules == 4:
            x = x.reshape(-1, 20)
        
        # apply our final output layer
        x = self.linear(x)

        return x
    
    
# 25214 parameters, 80% on CIFAR-10.
class CIFAR_CNN25K(nn.Module):
    def __init__(self):
        
        # parent constructor
        super().__init__()
        
        # first conv layer + pool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=80, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # second set of conv layers + pool + relu
        self.conv2 = nn.Conv2d(in_channels=80, out_channels=80, kernel_size=3, groups=80, padding=0)
        self.conv3 = nn.Conv2d(in_channels=80, out_channels=40, kernel_size=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        # third conv layer set.
        self.conv4 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)
        
        # fully-connected layer for classification.
        self.fc = nn.Linear(in_features=40, out_features=10)

    # encodes the forward-pass
    def forward(self, x):
        
        # first set
        x = self.conv1(x)
        x = self.pool1(x)
        x = F.relu(x)

        # second set
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = F.relu(x)

        # third set
        x = self.conv4(x)
        x = self.pool3(x)
        x = F.relu(x)

        # final classification.
        x = x.reshape(-1, 40)
        x = self.fc(x)

        return x
    
    
# 47650 parameters, 80% accuracy on CIFAR-10
class CIFAR_CNN47K(nn.Module):
    def __init__(self):
                
        # parent constructor
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=5, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=5, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.fc = nn.Linear(in_features=30, out_features=10)
        
    # encode the forward pass
    def forward(self, x):
        
        # first layer
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        
        # second layer
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        
        # third layer
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)

        # classification layer
        x = x.reshape(-1, 30)
        x = self.fc(x)

        return x
    
    
# 104778 parameters, 80% accuracy on CIFAR-10
class CIFAR_CNN100K(nn.Module):
    def __init__(self):
        super().__init__()
        
        # first layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.map1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.relu1 = nn.ReLU(inplace=True)

        # second layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.map2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.relu2 = nn.ReLU(inplace=True)

        # third layer
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.map3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.relu3 = nn.ReLU(inplace=True)
        
        # final fully connected layer
        self.fc = nn.Linear(in_features=128*3*3, out_features=10)
        
    def forward(self, x):
        
        # first layer
        x = self.conv1(x)
        x = self.map1(x)
        x = self.relu1(x)
        
        # second layer
        x = self.conv2(x)
        x = self.map2(x)
        x = self.relu2(x)
        
        # third layer
        x = self.conv3(x)
        x = self.map3(x)
        x = self.relu3(x)
        
        # classification layer
        x = x.reshape(-1, 128*3*3)
        x = self.fc(x)
        
        return x