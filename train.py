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
        nn.init.uniform_(m.weight, -10, 10)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

SAVE_IMAGE_EPOCH = 10
NUM_BATCHES_TO_SAVE = 1

def save_images(epoch, j, batch_index, image, boxes, conf_preds, det_preds, class_preds, labels, save_dir, matches):
    save_path = os.path.join(save_dir, f'gt_pred_{batch_index}_{j}.jpg')
    process_and_save_image(
        image, matches, gt_boxes=boxes.cpu(),
        confidences=conf_preds.cpu(),
        det_preds=det_preds.cpu().detach(),
        class_preds = class_preds,
        labels = labels,
        threshold=0.5,
        color_gt=(0, 255, 0),
        save_path=save_path
    )

def compute_batch_loss(batch, device, new_w, new_h, epoch, batch_index, images, selected_batches, save_dir):
    batch_loss = torch.tensor(0.0, device=device)
    batch_unmatched_loss = torch.tensor(0.0, device=device)
    batch_localization_loss = torch.tensor(0.0, device=device)
    batch_classifcation_loss = torch.tensor(0.0, device=device)
    
    for j, (boxes, labels, det_preds, conf_preds, class_preds, image) in enumerate(batch):
        loss, unmatched_loss, localization_loss, classification_loss, matches = custom_loss_function(det_preds, conf_preds, boxes, labels, class_preds, new_w, new_h)
        batch_loss += loss
        batch_unmatched_loss += unmatched_loss
        batch_localization_loss += localization_loss
        batch_classifcation_loss += classification_loss


        if epoch % SAVE_IMAGE_EPOCH == 0 and batch_index in selected_batches:
            save_images(epoch, j, batch_index, image, boxes, conf_preds, det_preds, class_preds, labels, save_dir, matches)

    return batch_loss, batch_unmatched_loss, batch_localization_loss, batch_classifcation_loss

def train_one_epoch(epoch, train_loader, model, optimizer, device, new_w, new_h, num_classes):
    model.train()

    total_loss = 0.0
    total_unmatched_loss =0
    total_localization_loss =0
    total_classification_loss =0

    save_dir = f'images/epoch_{epoch}'

    if epoch % SAVE_IMAGE_EPOCH == 0:
        print("Creating dir with predicted images.")
        os.makedirs(save_dir, exist_ok=True)
        #selected_batches = random.sample(range(0, 10), NUM_IMAGES_TO_SAVE)
        selected_batches = [0]
    else:
        selected_batches = []

    for i, (images, boxes_list, labels_list) in enumerate(train_loader):
        images = images.to(device)
        optimizer.zero_grad()
        detection_preds, confidence_preds, classification_preds = model(images)

        processed_batches = process_batches(
            boxes_list, labels_list, detection_preds, confidence_preds, classification_preds, device, new_w, new_h, epoch, i, images
        )

        #compute batch loss and save images
        batch_loss, batch_unmatched_loss, batch_localization_loss, batch_classifcation_loss  = compute_batch_loss(processed_batches, device, new_w, new_h, epoch, i, images, selected_batches, save_dir)
        batch_loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        total_unmatched_loss += batch_unmatched_loss.item()
        total_localization_loss += batch_localization_loss.item()
        total_classification_loss += batch_classifcation_loss.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_unmatched_loss = total_unmatched_loss / len(train_loader)
    avg_localization_loss = total_localization_loss / len(train_loader)
    avg_classification_loss = total_classification_loss / len(train_loader)
    
    return avg_train_loss, avg_unmatched_loss, avg_localization_loss, avg_classification_loss

def validate_one_epoch(epoch, val_loader, model, device, new_w, new_h, num_classes):
    model.eval()
    total_loss = 0.0
    total_unmatched_loss =0
    total_localization_loss =0
    total_classification_loss =0

    save_dir = f'images/val_epoch_{epoch}'

    if epoch % SAVE_IMAGE_EPOCH == 0:
        print("Creating dir with validation images.")
        os.makedirs(save_dir, exist_ok=True)
        selected_batches = [0]  # Modify as needed for validation
    else:
        selected_batches = []

    with torch.no_grad():
        for i, (images, boxes_list, labels_list) in enumerate(val_loader):
            images = images.to(device)
            detection_preds, confidence_preds, classification_preds = model(images)

            processed_batches = process_batches(
                boxes_list, labels_list, detection_preds, confidence_preds, classification_preds, device, new_w, new_h, epoch, i, images
            )

            batch_loss, batch_unmatched_loss, batch_localization_loss, batch_classifcation_loss = compute_batch_loss(
                processed_batches, device, new_w, new_h, epoch, i, images, selected_batches, save_dir
            )
            total_loss += batch_loss.item()
            total_unmatched_loss += batch_unmatched_loss.item()
            total_localization_loss += batch_localization_loss.item()
            total_classification_loss += batch_classifcation_loss.item()

    avg_val_loss = total_loss / len(val_loader)
    avg_unmatched_loss = total_unmatched_loss / len(val_loader)
    avg_localization_loss = total_localization_loss / len(val_loader)
    avg_classification_loss = total_classification_loss / len(val_loader)
    
    return avg_val_loss, avg_unmatched_loss, avg_localization_loss, avg_classification_loss


def manage_training(epoch, avg_val_loss, model, best_val_loss, epochs_no_improve, patience):
    stop_training = False

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'weights_max_1021.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Early stopping triggered at epoch {epoch}")
            stop_training = True  # Indicate to stop training

    if epoch % 20 == 0:
        torch.save(model.state_dict(), f'weights_max_{521+epoch}.pth')

    return best_val_loss, epochs_no_improve, stop_training


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_epochs = 1000
    learning_rate = 0.001
    batch_size = 10
    num_classes = 38
    max_images = 300
    num_boxes = 10
    best_val_loss = float('inf')
    new_w, new_h = 600, 300
    patience = 500     # Patience parameter for early stopping
    epochs_no_improve = 0

    # Initialize dataset and dataloader
    train_dataset = PascalVOCDataset(split='TRAIN', keep_difficult=False, max_images=max_images, new_w = new_w, new_h = new_h)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    val_dataset = PascalVOCDataset(split='VAL', keep_difficult=False, max_images=max_images, new_w = new_w, new_h = new_h)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    model = SimpleObjectDetector(num_boxes=num_boxes, num_classes=num_classes).to(device)
    model.det_head.apply(init_weights)

    #pretrained_weights = torch.load('weights_max_581.pth', map_location=device)
    # Update model's state_dict
    #model.load_state_dict(pretrained_weights, strict=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    for epoch in range(0, num_epochs):
        avg_train_loss, avg_unmatched_loss, avg_localization_loss, avg_classification_loss = train_one_epoch(epoch, train_loader, model, optimizer, device, new_w, new_h, num_classes)
        avg_val_loss, avg_unmatched_loss, avg_localization_loss, avg_classification_loss = validate_one_epoch(epoch, val_loader, model, device, new_w, new_h, num_classes)
        
        best_val_loss, epochs_no_improve, stop_training = manage_training(epoch, avg_val_loss, model, best_val_loss, epochs_no_improve, patience)
    
        print(epoch)

        if stop_training:
            break  # Exit the training loop

    print('Training and validation completed.')