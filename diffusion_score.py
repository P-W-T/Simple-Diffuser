import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Lambda
from PIL import Image
from timeit import default_timer as timer
from math import floor
from tqdm import tqdm
import copy
from scipy import linalg
from torch.utils.data import DataLoader, Dataset, TensorDataset

class fashionMNIST_CNN(nn.Module):
    """
    CNN to classify Fashion-MNIST samples
    """
    def __init__(self, img_size, n_layers=4, start_channels=3):
        """
        Initializes the Discriminator.
        args:
            - img_size (int): The width or height dimension of the images (assumes square images).
            - n_layers (int): The number of convolution layers in the model 
            (the image dimensions get smaller by a factor of 2 starting from the second layer).
            - start_channels (int): The number of channels in the first layer. 
        """
        super().__init__()
        self.img_size = img_size
        self.n_layers = n_layers
        self.start_channels = start_channels
        
        self.init_layer = nn.Conv2d(in_channels=1, out_channels=start_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.init_bn = nn.BatchNorm2d(num_features=start_channels)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels=max(start_channels, start_channels*2*i), out_channels=start_channels*2*(i+1), kernel_size=3, stride=2, padding=1, bias=False)
            for i in range(n_layers-1)
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(num_features=start_channels*2*(i+1))
            for i in range(n_layers-1)
        ])
        
        dim = img_size
        for i in range(n_layers-1):
            dim = floor((dim + 2.*1. - 1.*(3.-1.) - 1.)/2. + 1.)
            
        self.out_layer = nn.Linear(dim*dim*start_channels*2*(n_layers-1), 10)
    
    def forward(self, inputs, feature_extract=False):
        """
        Forward method for the classifier.
        args:
            - inputs (torch.tensor): Batch of images.
        """
        x = self.init_layer(inputs)        
        x = self.init_bn(x)        
        x = F.gelu(x)
        
        for i in range(self.n_layers-1):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = F.gelu(x)
            
        if feature_extract:
            return x
        
        return self.out_layer(x.view(x.shape[0], -1))


def evaluate_classifier(dataloader, classifier, device='cuda'):
    """
    Evaluate a classifier on testing data in the form of a dataloader.
    args:
        - dataloader (torch.utils.data.DataLoader): Dataloader contraining the data.
        - classifier(torch.nn.Module): The model to be evaluated.
        - device (str): The device for evaluation.
    """
    pred = []
    truth = []
    
    classifier.eval()
    classifier.to(device)
    
    for batch in dataloader:
        image, label = batch               
        image, label = image.to(device), label.to(device)

        with torch.no_grad():                        
            predictions = classifier(image).cpu()
            pred.append(predictions)
            truth.append(label.cpu())
    
    return pred, truth


def train_classifier(dataloader_train, dataloader_test, classifier, optimizer, epochs, device='cuda',
                        early_stop_patience=None, verbose=0):
    """
    Training method for a classifier.
    args:
        - dataloader_train (torch.utils.data.DataLoader): Dataloader contraining the training data.
        - dataloader_test (torch.utils.data.DataLoader): Dataloader contraining the testing or validation data. Can be set to None,
          in that case no evaluation will be done and early stopping will also not be performed.
        - classifier (torch.nn.Module): The model to be trained.
        - optimizer (torch.optim optimizer): The optimizer to update the model.
        - epochs (int): Number of epochs to train for.
        - device (str): The device for training.
        - early_stop_patience (int): The number of epochs to wait to stop training after imporvement on the testing data stops.
          If None, removes early stopping.
        - verbose (int): Determines the amount of information given during training with 0 being the least amount of information. 
    """
    loss_fn = nn.CrossEntropyLoss()
    classifier.to(device)
    classifier.train()
    
    patience = 0
    best_score = np.inf
    best_model = None
    
    length = len(dataloader_train)
    
    for epoch in range(epochs):
        print(f"Start training {epoch+1}/{epochs}")
        train_loss = 0
        if verbose > 0:              
            start = timer()
        for batch in tqdm(dataloader_train, total=length, disable=(verbose<1)):            
            image, label = batch               
            image, label = image.to(device), label.to(device)
            predictions = torch.squeeze(classifier(image))
            loss = loss_fn(predictions, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().detach()
        
        if dataloader_test is not None:            
            pred, truth = evaluate_classifier(dataloader_test, classifier, device)
            accuracy = torch.sum(torch.argmax(nn.functional.softmax(torch.cat(pred, dim=0), dim=-1), dim=-1)==torch.cat(truth, dim=0))/len(torch.cat(truth, dim=0))
            classifier.train()
            
            if early_stop_patience is not None:
                score = loss_fn(torch.cat(pred, dim=0), torch.cat(truth, dim=0)).item()
                if score < best_score:
                    best_score = score
                    patience = 0
                    best_model = copy.deepcopy(classifier)
                else:
                    patience += 1
        
        if verbose > 0:      
            end = timer()
            current_time = end - start
            minutes = int(current_time/60)
            seconds = current_time - minutes*60
            if dataloader_test is not None:
                print(f"Epoch {epoch+1}/{epochs} - training loss: {train_loss/length:.4f} - Accuracy: {accuracy} - {minutes}m {seconds:2f}s")
            else:
                print(f"Epoch {epoch+1}/{epochs} - training loss: {train_loss/length:.4f} - {minutes}m {seconds:2f}s")
            
        if early_stop_patience is not None:
            if patience > early_stop_patience:
                return best_model
    return classifier


def inception_score(dataloader, model, splits=10, contains_labels=False, device='cuda'):
    """
    Calculates the inception score for images in the dataloader. 
    args:
        - dataloader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate.
        - model (torch.nn.Module): The classification model.
        - splits (int): Number of splits to calculate the score over.
        - contains_labels (bool): Signifies whether dataloader contains labels or not
        - device (str): The device for evaluation.
    """
    model.eval()  # Ensure the model is in evaluation mode.
    model.to(device)
    preds = []

    # Loop through the dataset
    with torch.no_grad():  # Disable gradient computation for efficiency.
        for images in dataloader:
            
            if contains_labels:
                images, _ = images
            images = images.to(device)
            outputs = model(images)  # Get model predictions (class probabilities).
            preds.append(outputs.detach().cpu())  # Store predictions.

    preds = nn.functional.softmax(torch.cat(preds, dim=0), dim=-1).numpy() # Combine predictions from all batches.

    # Calculate the score for each split
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        # Calculate the KL divergence for each split and then the exponential of its mean.
        kl_div = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl_div = np.mean(np.sum(kl_div, 1))
        scores.append(np.exp(kl_div))

    # Return the mean and standard deviation of the calculated scores.
    if splits == 1:
        return scores[0]
    return np.mean(scores), np.std(scores)

def get_feature_vectors(dataloader, model, contains_labels=False, device='cuda'):
    """
    Gets the feature vectors for images in a dataloader from a compatible model.
    args:
        - dataloader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate.
        - model (torch.nn.Module): The classification model. The models needs an additional input in the forward 
          method 'feature_extract' that allows the extraction of an intermediate outputs.
        - contains_labels (bool): Signifies whether dataloader contains labels or not
        - device (str): The device for evaluation.
    """
    model.eval()
    model.to(device)
    feature_vectors = []
    
    with torch.no_grad():
        for images in dataloader:
            if contains_labels:
                images, _ = images
            images = images.to(device)
            features = model(images, feature_extract=True)
            feature_vectors.append(features.cpu())
            
    feature_vectors = torch.cat(feature_vectors, dim=0)
    feature_vectors = torch.mean(feature_vectors.view(feature_vectors.shape[0], feature_vectors.shape[1], -1), dim=-1).numpy()
    
    return feature_vectors


def calculate_fid(real_features, gen_features):
    """
    Calculates the Fréchet inception distance (FID) score for images for whom the features where extracted. 
    args:
        - real_features (np.array): The extracted features from real data.
        - gen_features (np.array): The extracted features from generated data.
    """
    # Calculate the mean and covariance of both sets
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

    # Calculate the squared difference in means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Calculate the sqrt of the product of covariances
    covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]

    # Check for complex numbers and take the real part
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Calculate the FID score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


class TensorImagesDataset(Dataset):
    def __init__(self, tensor_images, transforms=None):
        """
        Args:
            tensor_images (torch.Tensor): A list of images as tensors.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.tensor_images = tensor_images
        self.transforms = transforms

    def __len__(self):
        return len(self.tensor_images)

    def __getitem__(self, idx):
        image = self.tensor_images[idx]

        if self.transforms:
            image = self.transforms(image)

        return image