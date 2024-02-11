import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
import json

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

SAVE_IMAGE_EPOCH = 2
NUM_BATCHES_TO_SAVE = 1


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
        nn.init.uniform_(m.weight, -15, 15)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def save_images(epoch, j, batch_index, image, boxes, det_preds, class_preds, labels, save_dir, matches):
    save_path = os.path.join(save_dir, f'gt_pred_{batch_index}_{j}.jpg')
    process_and_save_image(
        image, matches, gt_boxes=boxes.cpu(),
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
    #batch_confidence_loss = torch.tensor(0.0, device=device)
    
    for j, (boxes, labels, det_preds, class_preds, image) in enumerate(batch):
        loss, unmatched_loss, localization_loss, classification_loss, matches = custom_loss_function(epoch, det_preds, boxes, labels, class_preds, new_w, new_h)
        batch_loss += loss
        batch_unmatched_loss += unmatched_loss
        batch_localization_loss += localization_loss
        batch_classifcation_loss += classification_loss

        if epoch % SAVE_IMAGE_EPOCH == 0 and batch_index in selected_batches:
            save_images(epoch, j, batch_index, image, boxes, det_preds, class_preds, labels, save_dir, matches)

    return batch_loss, batch_unmatched_loss, batch_localization_loss, batch_classifcation_loss

def train_one_epoch(epoch, train_loader, model, optimizer, device, new_w, new_h, num_classes, run_dir):
    model.train()

    total_loss = 0.0
    total_unmatched_loss =0
    total_localization_loss =0
    total_classification_loss =0
    save_dir = os.path.join(run_dir,f'images/epoch_{epoch}')

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
        detection_preds, classification_preds = model(images)

        processed_batches = process_batches(
            boxes_list, labels_list, detection_preds, classification_preds, device, new_w, new_h, epoch, i, images
        )

        #compute batch loss and save images
        batch_loss, batch_unmatched_loss, batch_localization_loss, batch_classifcation_loss  = compute_batch_loss(processed_batches, device, new_w, new_h, epoch, i, images, selected_batches, save_dir)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        total_unmatched_loss += batch_unmatched_loss.item()
        total_localization_loss += batch_localization_loss.item()
        total_classification_loss += batch_classifcation_loss.item()
        #total_confidence_loss += batch_confidence_loss.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_unmatched_loss = total_unmatched_loss / len(train_loader)
    avg_localization_loss = total_localization_loss / len(train_loader)
    avg_classification_loss = total_classification_loss / len(train_loader)
    #avg_confidence_loss = total_confidence_loss / len(train_loader)
    
    return avg_train_loss, avg_unmatched_loss, avg_localization_loss, avg_classification_loss

def validate_one_epoch(epoch, val_loader, model, device, new_w, new_h, num_classes, run_dir):
    model.eval()
    total_loss = 0.0
    total_unmatched_loss =0
    total_localization_loss =0
    total_classification_loss =0

    save_dir = os.path.join(run_dir,f'images/val_epoch_{epoch}')

    if epoch % SAVE_IMAGE_EPOCH == 0:
        print("Creating dir with validation images.")
        os.makedirs(save_dir, exist_ok=True)
        selected_batches = [0]  # Modify as needed for validation
    else:
        selected_batches = []

    with torch.no_grad():
        for i, (images, boxes_list, labels_list) in enumerate(val_loader):
            images = images.to(device)
            detection_preds, classification_preds = model(images)

            processed_batches = process_batches(
                boxes_list, labels_list, detection_preds, 
                 classification_preds, device, new_w, new_h, epoch, i, images
            )

            batch_loss, batch_unmatched_loss, batch_localization_loss, batch_classifcation_loss = compute_batch_loss(
                processed_batches, device, new_w, new_h, epoch, i, images, selected_batches, save_dir
            )
            total_loss += batch_loss.item()
            total_unmatched_loss += batch_unmatched_loss.item()
            total_localization_loss += batch_localization_loss.item()
            total_classification_loss += batch_classifcation_loss.item()
            #total_confidence_loss += batch_confidence_loss.item()

    avg_val_loss = total_loss / len(val_loader)
    avg_unmatched_loss = total_unmatched_loss / len(val_loader)
    avg_localization_loss = total_localization_loss / len(val_loader)
    avg_classification_loss = total_classification_loss / len(val_loader)
    #avg_confidence_loss = total_confidence_loss / len(val_loader)
    
    return avg_val_loss, avg_unmatched_loss, avg_localization_loss, avg_classification_loss


def manage_training(epoch, avg_val_loss, model, best_val_loss, epochs_no_improve, patience, run_dir):
    stop_training = False

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(run_dir,'best.pth'))
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Early stopping triggered at epoch {epoch}")
            stop_training = True  # Indicate to stop training

    if epoch % 20 == 0:
        torch.save(model.state_dict(), os.path.join(run_dir,f'weights_{epoch}.pth'))

    return best_val_loss, epochs_no_improve, stop_training

def plot_and_save_losses(epoch, train_loss, val_loss, train_loc_loss, val_loc_loss, train_class_loss, val_class_loss, train_unmatched_loss, val_unmatched_loss, num_epochs, save_dir='loss_graphs'):
    """
    Update to plot and save training and validation loss graphs every 10 epochs.
    :param epoch: Current epoch number.
    :param train_loss, val_loss: Total training and validation losses.
    :param train_loc_loss, val_loc_loss: Localization losses for training and validation.
    :param train_class_loss, val_class_loss: Classification losses for training and validation.
    :param train_unmatched_loss, val_unmatched_loss: Confidence/unmatched losses for training and validation.
    :param num_epochs: Total number of epochs for the training.
    :param save_dir: Directory to save the loss graphs.
    """
    # Append current epoch losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_loc_losses.append(train_loc_loss)
    val_loc_losses.append(val_loc_loss)
    train_class_losses.append(train_class_loss)
    val_class_losses.append(val_class_loss)
    train_unmatched_losses.append(train_unmatched_loss)
    val_unmatched_losses.append(val_unmatched_loss)

    # Plot and save every 10 epochs or after the last epoch
    if (epoch + 1) % 1 == 0 or (epoch + 1) == num_epochs:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Plot training losses
        plt.figure(figsize=(10, 7))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(train_loc_losses, label='Localization Loss')
        plt.plot(train_class_losses, label='Classification Loss')
        plt.plot(train_unmatched_losses, label='Unmatched Loss')
        #plt.plot(train_confidence_losses, label='Confidence Loss')
        plt.title('Training Losses through Epoch {}'.format(epoch + 1))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{save_dir}/training_losses.png')
        plt.close()

        # Plot validation losses
        plt.figure(figsize=(10, 7))
        plt.plot(val_losses, label='Validation Loss')
        plt.plot(val_loc_losses, label='Localization Loss')
        plt.plot(val_class_losses, label='Classification Loss')
        plt.plot(val_unmatched_losses, label='Unmatched Loss')
        #plt.plot(val_confidence_losses, label='Condidence Loss')
        plt.title('Validation Losses through Epoch {}'.format(epoch + 1))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{save_dir}/validation_losses.png')
        plt.close()

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a unique directory name
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join('runs', run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    train_losses, val_losses = [], []
    train_loc_losses, val_loc_losses = [], []
    train_class_losses, val_class_losses = [], []
    train_unmatched_losses, val_unmatched_losses = [], []
    train_confidence_losses, val_confidence_losses = [], []

    # Hyperparameters
    num_epochs = 1000
    learning_rate = 0.00001
    batch_size = 10
    num_classes = 5+1
    num_boxes = 10
    best_val_loss = float('inf')
    new_w, new_h = 600, 300
    patience = 500     # Patience parameter for early stopping
    epochs_no_improve = 0


    hyperparameters = {
    'num_epochs': num_epochs,
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'num_classes': num_classes,
    'num_boxes': num_boxes,
    'new_width': new_w,
    'new_height': new_h,
    'patience': patience,
    }

    # Saving hyperparameters as a JSON file
    with open(os.path.join(run_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    # Initialize dataset and dataloader
    train_dataset = PascalVOCDataset(split='TRAIN', keep_difficult=False, max_images=30, new_w = new_w, new_h = new_h)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    val_dataset = PascalVOCDataset(split='VAL', keep_difficult=False, max_images=30, new_w = new_w, new_h = new_h)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

    model = SimpleObjectDetector(num_boxes=num_boxes, num_classes=num_classes).to(device)
    model.det_head.apply(init_weights)

    #pretrained_weights = torch.load('weights_max_581.pth', map_location=device)
    #model.load_state_dict(pretrained_weights, strict=False)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    #scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    for epoch in range(0, num_epochs):
        avg_train_loss, avg_unmatched_loss_t, avg_localization_loss_t, avg_classification_loss_t = train_one_epoch(epoch, train_loader, model, optimizer, device, new_w, new_h, num_classes, run_dir)
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss}, unmatched: {avg_unmatched_loss_t}, localization {avg_localization_loss_t}, classification {avg_classification_loss_t}")
        
        avg_val_loss, avg_unmatched_loss, avg_localization_loss, avg_classification_loss = validate_one_epoch(epoch, val_loader, model, device, new_w, new_h, num_classes, run_dir)
        print(f"Epoch {epoch}: Validation Loss: {avg_val_loss}, unmatched: {avg_unmatched_loss}, localization {avg_localization_loss}, classification {avg_classification_loss}")

        best_val_loss, epochs_no_improve, stop_training = manage_training(epoch, avg_val_loss, model, best_val_loss, epochs_no_improve, patience, run_dir)
    
        plot_and_save_losses(
        epoch,
        avg_train_loss, avg_val_loss,
        avg_localization_loss_t, avg_localization_loss,  # Assuming these are available
        avg_classification_loss_t, avg_classification_loss,  # Same assumption
        avg_unmatched_loss_t, avg_unmatched_loss,  # Assuming these are tracked
        num_epochs,
        run_dir)

        if stop_training:
            break  # Exit the training loop