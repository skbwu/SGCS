import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, copy, os, shutil
from tqdm.notebook import tqdm
from IPython.display import clear_output
from sklearn.manifold import TSNE
import seaborn as sns

# for SmoothGrad saliency maps (DO NOT USE MAGNITUDE!)
from gradients import SmoothGrad

# for loading datasets
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

# custom utilities + optimized resnets
import utils
from resnet import resnet20, resnet32, resnet44

# no fancy tricks -- let's keep it simple
train_transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
batch_size = 256

# create a directory for our figures
if "figures" not in os.listdir():
    os.mkdir("figures")
    
# subdirectory for our qualitative figures
if "qualitative" not in os.listdir("figures"):
    os.mkdir("figures/qualitative")
    
# use a GPU if possible
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
# friendly colors
colors = ['#377eb8', '#ff7f00', '#4daf4a',
          '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00']


# load our training and test data
data_train = MNIST(root="./data", train=True, download=True, transform=train_transforms)
data_test = MNIST(root="./data", train=False, download=True, transform=train_transforms)

# load in our log files too
train_scores = pd.read_csv("logs/mnist_train_scores.csv")
test_scores = pd.read_csv("logs/mnist_test_scores.csv")

# compute the difficulties of each train and test point
train_difficulties = train_scores[train_scores.columns[4:]].mean(axis=0).values
test_difficulties = test_scores[test_scores.columns[4:]].mean(axis=0).values


# what dataset are we working with?
dataset = "MNIST"

# let's look at hardest, hard, easier, and easiest images.
for quantile in [0.25, 0.50, 0.75, 1.00]:

    # go thru each of our 10 classes
    for class_label in range(10):

        # go thru our train + test splits
        for split in ["train", "test"]:
            
            # pick the image that we will work with for this figure
            if split == "train":

                # query all data points + difficulties that correspond to this class
                class_idxs = np.argwhere(data_train.targets.numpy() == class_label).flatten()
                class_train_difficulties = train_difficulties[class_idxs]

                # what's the accuracy cutoff for this threshold?
                critical_val = np.quantile(np.unique(class_train_difficulties), q=quantile)

                # which datapoint are we generating a figure
                class_viz_idx = class_idxs[np.abs(class_train_difficulties - critical_val).argmin()]
                img = data_train[class_viz_idx][0]
            
            # ... go for the test set
            else:
                
                # query all data points + difficulties that correspond to this class
                class_idxs = np.argwhere(data_test.targets.numpy() == class_label).flatten()
                class_test_difficulties = test_difficulties[class_idxs]

                # what's the accuracy cutoff for this threshold?
                critical_val = np.quantile(np.unique(class_test_difficulties), q=quantile)

                # which datapoint are we generating a figure
                class_viz_idx = class_idxs[np.abs(class_test_difficulties - critical_val).argmin()]
                img = data_test[class_viz_idx][0]
                
            
            ######### LOGISTICS
            
            # create our foldername + make the requisite folder
            foldername = f"{dataset}_class={class_label}_split={split}_diff-q={quantile}"
            if foldername not in os.listdir("figures/qualitative"):
                os.mkdir(f"figures/qualitative/{foldername}")
                
            # visualize the image itself
            fig, ax = plt.subplots(dpi=200)
            ax.imshow(img.numpy()[0])
            ax.axis("off")
            ax.set_title(f"Class {class_label}, % Correct: {np.round(critical_val, 3)}", fontsize=20)
            plt.tight_layout()
            plt.savefig(f"figures/qualitative/{foldername}/img.png", facecolor="white", bbox_inches="tight")
            plt.close()
            
            # now move the image to gpu
            img = img.reshape(1, *img.shape)
            img = img.to(device)
            
            
            ######### FIGURE 1
            
            # create a 9 x 3 grid of subplots
            fig, ax = plt.subplots(3, 9, dpi=200, figsize=(9, 3))

            # go thru all 9 possible models
            for j, model_desc in enumerate(["mlp_num-layers=1", "mlp_num-layers=2", "mlp_num-layers=3",
                                            "cnn_num-modules=2", "cnn_num-modules=3", "cnn_num-modules=4",
                                            "resnet_variant=20", "resnet_variant=32", "resnet_variant=44"]):

                # load in the base model based on the specific settings + also the weights
                if "mlp" in model_desc:
                    variant = int(model_desc.split("=")[1])
                    model = utils.MNIST_MLP(num_layers=variant, data_dim=1024)
                elif "cnn" in model_desc:
                    variant = int(model_desc.split("=")[1])
                    model = utils.MNIST_CNN(num_modules=variant)
                elif "resnet" in model_desc:
                    variant = int(model_desc.split("=")[1])
                    if variant == 20:
                        model = resnet20()
                    elif variant == 32:
                        model = resnet32()
                    elif variant == 44:
                        model = resnet44()

                # track how many got it correct
                symbs = ""

                # go thru 3 seeds apiece
                for i in range(3):

                    # load in the correct weights + set to eval mode
                    model.load_state_dict(torch.load(f"models/MNIST/{model_desc}_seed={str(i).zfill(3)}/099.pth"))
                    model.eval(); model.to(device)

                    # compute our SmoothGrad saliency map
                    smooth_grad = SmoothGrad(pretrained_model=model, cuda=True, n_samples=50, magnitude=False)

                    # special case for resnet!
                    if "resnet" in model_desc:
                        saliency_map = smooth_grad(torch.cat([img, img, img], dim=1), index=None)
                        if model(torch.cat([img, img, img], dim=1)).argmax().item() == class_label:
                            correct = True; symb = "✓"
                        else:
                            correct = False; symb = "✗"
                    else:
                        saliency_map = smooth_grad(img, index=None)
                        if model(img).argmax().item() == class_label:
                            correct = True; symb = "✓"
                        else:
                            correct = False; symb = "✗"

                    # add the symbol to our string
                    symbs += symb

                    # make saliency map 2d, if not already
                    saliency_map_2d = np.sum(saliency_map, axis=0)

                    # show our saliency map
                    ax[i, j].imshow(saliency_map_2d, cmap="viridis")
                    ax[i, j].set_xticks([])
                    ax[i, j].set_yticks([])

                    # beautify accordingly
                    if j == 0:
                        ax[i, j].set_ylabel(f"Seed {i}", fontsize=8)

                # at the end, let's do our incrementing
                model_header = model_desc.split("_")[0].capitalize()
                ax[0, j].set_title(f"{model_header.upper()}-{variant} ({symbs})", fontsize=6)

            # beautify at the very end
            plt.tight_layout()
            plt.savefig(f"figures/qualitative/{foldername}/maps.png", facecolor="white", bbox_inches="tight")
            plt.close()
            
            
            ######### FIGURE 2
            
            # get 50 saliency maps for extreme MLP, CNN, and resnet - create a roster of colors
            color_codes = ([0] * 50) + ([1] * 50) + ([2] * 50) + ([3] * 50) + ([4] * 50) + ([5] * 50)

            # create an array to store all of our saliency maps (flattened)
            X_saliency = []

            # MLP no. 1
            model = utils.MNIST_MLP(num_layers=1, data_dim=1024); model.to(device); model.eval()
            for i in range(50):

                # load in the weights of that iteration
                model.load_state_dict(torch.load(f"models/MNIST/mlp_num-layers=1_seed={str(i).zfill(3)}/099.pth"))

                # compute our SmoothGrad saliency map + add it to our list
                smooth_grad = SmoothGrad(pretrained_model=model, cuda=True, n_samples=50, magnitude=False)
                saliency_map_2d = np.sum(smooth_grad(img, index=None), axis=0)
                X_saliency.append(saliency_map_2d.flatten())

            # MLP no. 2
            model = utils.MNIST_MLP(num_layers=3, data_dim=1024); model.to(device); model.eval()
            for i in range(50):

                # load in the weights of that iteration
                model.load_state_dict(torch.load(f"models/MNIST/mlp_num-layers=3_seed={str(i).zfill(3)}/099.pth"))

                # compute our SmoothGrad saliency map + add it to our list
                smooth_grad = SmoothGrad(pretrained_model=model, cuda=True, n_samples=50, magnitude=False)
                saliency_map_2d = np.sum(smooth_grad(img, index=None), axis=0)
                X_saliency.append(saliency_map_2d.flatten())

            # CNN no. 1
            model = utils.MNIST_CNN(num_modules=2); model.to(device); model.eval()
            for i in range(50):

                # load in the weights of that iteration
                model.load_state_dict(torch.load(f"models/MNIST/cnn_num-modules=2_seed={str(i).zfill(3)}/099.pth"))

                # compute our SmoothGrad saliency map + add it to our list
                smooth_grad = SmoothGrad(pretrained_model=model, cuda=True, n_samples=50, magnitude=False)
                saliency_map_2d = np.sum(smooth_grad(img, index=None), axis=0)
                X_saliency.append(saliency_map_2d.flatten())

            # CNN no. 2
            model = utils.MNIST_CNN(num_modules=4); model.to(device); model.eval()
            for i in range(50):

                # load in the weights of that iteration
                model.load_state_dict(torch.load(f"models/MNIST/cnn_num-modules=4_seed={str(i).zfill(3)}/099.pth"))

                # compute our SmoothGrad saliency map + add it to our list
                smooth_grad = SmoothGrad(pretrained_model=model, cuda=True, n_samples=50, magnitude=False)
                saliency_map_2d = np.sum(smooth_grad(img, index=None), axis=0)
                X_saliency.append(saliency_map_2d.flatten())

            # resnet20
            model = resnet20(); model.to(device); model.eval()
            for i in range(50):

                # load in the weights of that iteration
                model.load_state_dict(torch.load(f"models/MNIST/resnet_variant=20_seed={str(i).zfill(3)}/099.pth"))

                # compute our SmoothGrad saliency map + add it to our list
                smooth_grad = SmoothGrad(pretrained_model=model, cuda=True, n_samples=50, magnitude=False)
                saliency_map_2d = np.sum(smooth_grad(torch.cat([img, img, img], dim=1), index=None), axis=0)
                X_saliency.append(saliency_map_2d.flatten())

            # resnet44
            model = resnet44(); model.to(device); model.eval()
            for i in range(50):

                # load in the weights of that iteration
                model.load_state_dict(torch.load(f"models/MNIST/resnet_variant=44_seed={str(i).zfill(3)}/099.pth"))

                # compute our SmoothGrad saliency map + add it to our list
                smooth_grad = SmoothGrad(pretrained_model=model, cuda=True, n_samples=50, magnitude=False)
                saliency_map_2d = np.sum(smooth_grad(torch.cat([img, img, img], dim=1), index=None), axis=0)
                X_saliency.append(saliency_map_2d.flatten())

            # compute our t-SNE
            X_saliency = np.array(X_saliency)
            tsne = TSNE(n_components=2, random_state=858)
            X_trans = tsne.fit_transform(X_saliency)

            # create our figure
            fig, ax = plt.subplots(dpi=200)
            plt.scatter(X_trans[:,0], X_trans[:,1], c=[colors[color_code] for color_code in color_codes])
            plt.scatter([np.nan], [np.nan], label="MLP-1", color=colors[0])
            plt.scatter([np.nan], [np.nan], label="MLP-3", color=colors[1])
            plt.scatter([np.nan], [np.nan], label="CNN-2", color=colors[2])
            plt.scatter([np.nan], [np.nan], label="CNN-4", color=colors[3])
            plt.scatter([np.nan], [np.nan], label="RESNET-20", color=colors[4])
            plt.scatter([np.nan], [np.nan], label="RESNET-44", color=colors[5])
            plt.grid()
            plt.legend(fontsize=12)
            plt.xlabel("T-SNE Component 1")
            plt.ylabel("T-SNE Component 2")
            plt.tight_layout()
            plt.savefig(f"figures/qualitative/{foldername}/tsne.png", facecolor="white", bbox_inches="tight")
            plt.close()
    
            ######### FIGURE 3
        
            # get the largest MLP, CNN, and ResNet saliencies
            mlp_sals, cnn_sals, res_sals = X_saliency[50:100], X_saliency[150:200], X_saliency[250:300]

            # compute cosine similarities of MLP only
            cossim_mlp = mlp_sals @ mlp_sals.T
            cossim_mlp /= np.sqrt((mlp_sals ** 2).sum(axis=1)).reshape(-1, 1)
            cossim_mlp /= np.sqrt((mlp_sals ** 2).sum(axis=1)).reshape(1, -1)
            np.fill_diagonal(cossim_mlp, np.nan)
            cossim_mlp = cossim_mlp.flatten()[~np.isnan(cossim_mlp.flatten())]

            # compute cosine similarities of CNN only
            cossim_cnn = cnn_sals @ cnn_sals.T
            cossim_cnn /= np.sqrt((cnn_sals ** 2).sum(axis=1)).reshape(-1, 1)
            cossim_cnn /= np.sqrt((cnn_sals ** 2).sum(axis=1)).reshape(1, -1)
            np.fill_diagonal(cossim_cnn, np.nan)
            cossim_cnn = cossim_cnn.flatten()[~np.isnan(cossim_cnn.flatten())]

            # compute cosine similarities of RESNET only
            cossim_res = res_sals @ res_sals.T
            cossim_res /= np.sqrt((res_sals ** 2).sum(axis=1)).reshape(-1, 1)
            cossim_res /= np.sqrt((res_sals ** 2).sum(axis=1)).reshape(1, -1)
            np.fill_diagonal(cossim_res, np.nan)
            cossim_res = cossim_res.flatten()[~np.isnan(cossim_res.flatten())]

            # compute cosine similarities of MLP + CNNs
            cossim_mlp_cnn = mlp_sals @ cnn_sals.T
            cossim_mlp_cnn /= np.sqrt((mlp_sals ** 2).sum(axis=1)).reshape(-1, 1)
            cossim_mlp_cnn /= np.sqrt((cnn_sals ** 2).sum(axis=1)).reshape(1, -1)

            # compute cosine similarities of MLP + RESNET
            cossim_mlp_res = mlp_sals @ res_sals.T
            cossim_mlp_res /= np.sqrt((mlp_sals ** 2).sum(axis=1)).reshape(-1, 1)
            cossim_mlp_res /= np.sqrt((res_sals ** 2).sum(axis=1)).reshape(1, -1)

            # compute cosine similarities of CNN + RESNET
            cossim_cnn_res = cnn_sals @ res_sals.T
            cossim_cnn_res /= np.sqrt((cnn_sals ** 2).sum(axis=1)).reshape(-1, 1)
            cossim_cnn_res /= np.sqrt((res_sals ** 2).sum(axis=1)).reshape(1, -1)

            # create three subplots here
            fig, ax = plt.subplots(1, 3, dpi=200, figsize=(20, 5))

            # MLP vs. CNN
            sns.kdeplot(cossim_mlp.flatten(), ax=ax[0], color=colors[0], label="MLP-3")
            sns.kdeplot(cossim_cnn.flatten(), ax=ax[0], color=colors[1], label="CNN-4")
            sns.kdeplot(cossim_mlp_cnn.flatten(), ax=ax[0], color=colors[2], label="Between")
            ax[0].legend(fontsize=14)
            ax[0].set_xlabel("Cosine Similarity", fontsize=20)
            ax[0].set_ylabel("Density", fontsize=16)
            ax[0].set_title("MLP-3 and CNN-4", fontsize=20)
            ax[0].grid()

            # MLP vs. RESNET
            sns.kdeplot(cossim_mlp.flatten(), ax=ax[1], color=colors[0], label="MLP-3")
            sns.kdeplot(cossim_res.flatten(), ax=ax[1], color=colors[1], label="RESNET-44")
            sns.kdeplot(cossim_mlp_res.flatten(), ax=ax[1], color=colors[2], label="Between")
            ax[1].legend(fontsize=14)
            ax[1].set_xlabel("Cosine Similarity", fontsize=20)
            ax[1].set_ylabel("")
            ax[1].set_title("MLP-3 and RESNET-44", fontsize=20)
            ax[1].grid()

            # CNN vs. RESNET
            sns.kdeplot(cossim_cnn.flatten(), ax=ax[2], color=colors[0], label="CNN-4")
            sns.kdeplot(cossim_res.flatten(), ax=ax[2], color=colors[1], label="RESNET-44")
            sns.kdeplot(cossim_cnn_res.flatten(), ax=ax[2], color=colors[2], label="Between")
            ax[2].legend(fontsize=14)
            ax[2].set_xlabel("Cosine Similarity", fontsize=20)
            ax[2].set_ylabel("")
            ax[2].set_title("CNN-4 and RESNET-44", fontsize=20)
            ax[2].grid()

            plt.tight_layout()
            plt.savefig(f"figures/qualitative/{foldername}/kde.png", facecolor="white", bbox_inches="tight")
            plt.close()