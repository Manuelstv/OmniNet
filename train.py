import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import PascalVOCDataset
from foviou import *
from model import (SimpleObjectDetector, SimpleObjectDetectorMobile,
                   SimpleObjectDetectorResnet)
from plot_tools import process_and_save_image
from sphiou import Sph
from losses import *
from utils import *
from torch.optim.lr_scheduler import StepLR
import os
import random


def init_weights(m):
    """
    Initialize the weights of a linear layer.

    Args:
    - m (nn.Module): A linear layer of a neural network.

    Note:
    - This function is designed to be applied to a linear layer of a PyTorch model.
    - If the layer has a bias term, it will be initialized to zero.
    """
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -1, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def train_one_epoch(epoch, train_loader, model, optimizer, device, new_w, new_h, num_classes):
    """
    Train the model for one epoch using Mean Squared Error (MSE) as the loss function.

    This function iterates over the training data loader, performs forward and backward
    passes, and updates the model parameters. Additionally, it saves images for visualization
    and calculates the average training loss for the epoch.

    Parameters:
    - epoch (int): The current epoch number.
    - train_loader (DataLoader): The DataLoader object providing the training data.
    - model (nn.Module): The neural network model being trained.
    - optimizer (Optimizer): The optimizer used for updating model parameters.
    - device (torch.device): The device (CPU/GPU) on which the computations are performed.
    - new_w (int): The new width dimension for the image after processing.
    - new_h (int): The new height dimension for the image after processing.

    Returns:
    - None
    """

    model.train()
    total_loss = 0.0

    #loop over dataset; each iteration "is" a batch
    for i, (images, boxes_list, labels_list) in enumerate(train_loader):
        images = images.to(device)
        optimizer.zero_grad()
        detection_preds, confidence_preds, classification_preds = model(images)

        batch_loss = torch.tensor(0.0, device=device)        

        #loop over one batch; each iteration "is" an image
        for boxes, labels, det_preds, conf_preds, class_preds, image in process_batches(boxes_list, labels_list, detection_preds, confidence_preds, classification_preds, device, new_w, new_h, epoch, i, images):
            loss, matches = custom_loss_function(det_preds, conf_preds, boxes, labels, class_preds, new_w, new_h)
            print(class_preds)
            print(labels)
            batch_loss += loss

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        #scheduler.step()
        total_loss += batch_loss.item()

    #melhorar organizacao disso (so precisa plotar uma vex)
    process_and_save_image(image,
                               matches, 
                       gt_boxes=boxes.cpu(),
                       confidences = conf_preds.cpu(), 
                       det_preds=det_preds.cpu().detach(), 
                       threshold=0.5, 
                       color_gt=(0, 255, 0), 
                       save_path=f'/home/mstveras/images5/gt_pred_{epoch}.jpg')

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: Train Loss: {avg_train_loss}")

if __name__ == "__main__":

    torch.cuda.empty_cache()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_epochs = 1000
    learning_rate = 0.0001
    batch_size = 10
    num_classes = 38
    max_images = 300
    num_boxes = 10
    best_val_loss = float('inf')
    new_w, new_h = 600, 300

    # Initialize dataset and dataloader
    train_dataset = PascalVOCDataset(split='TRAIN', keep_difficult=False, max_images=max_images, new_w = new_w, new_h = new_h)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    #val_dataset = PascalVOCDataset(split='VAL', keep_difficult=False, max_images=max_images, new_w = new_w, new_h = new_h)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    model = SimpleObjectDetector(num_boxes=num_boxes, num_classes=num_classes).to(device)
    model.det_head.apply(init_weights)

    pretrained_weights = torch.load('weights_501.pth', map_location=device)
    # Update model's state_dict
    model.load_state_dict(pretrained_weights, strict=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(num_epochs):
        train_one_epoch(epoch, train_loader, model, optimizer, device, new_w, new_h, num_classes)
        #validate_one_epoch_mse(epoch, val_loader, model, device, new_w, new_h)
        if epoch%20==0:
            torch.save(model.state_dict(), f'weights_max_{521+epoch}.pth')

    print('Training and validation completed.')