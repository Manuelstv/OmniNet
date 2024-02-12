import torch
from foviou import *
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import pdb



def iou_matching(gt_boxes_in, pred_boxes_in, new_w, new_h, iou_threshold):
    """
    Match ground truth and predicted boxes based on IoU scores, simply assigning a match if IoU > 0.5.
    This function does not use Hungarian matching for initial pair selection.
    
    Args:
    - gt_boxes_in (Tensor): A tensor of ground truth bounding boxes.
    - pred_boxes_in (Tensor): A tensor of predicted bounding boxes.
    
    Returns:
    - list of tuples: Matched pairs of ground truth and predicted boxes with IoU > 0.5.
    - Tensor: IoU scores for the matched pairs.
    """
    if gt_boxes_in.size(0) == 0 or pred_boxes_in.size(0) == 0:
        return [], None
    
    gt_boxes = gt_boxes_in.clone().to(torch.float)
    pred_boxes = pred_boxes_in.clone().to(torch.float)
    
    # Compute the IoU matrix
    iou_matrix = fov_iou_batch(gt_boxes, pred_boxes, new_w, new_h)
    
    # Find matches where IoU > 0.5
    matched_pairs = []
    for gt_idx in range(gt_boxes.size(0)):
        for pred_idx in range(pred_boxes.size(0)):
            if iou_matrix[gt_idx, pred_idx] > iou_threshold:
                matched_pairs.append((gt_idx, pred_idx))
    
    # No need to convert IoUs to cost or use linear_sum_assignment from scipy
    
    return matched_pairs, iou_matrix

def hungarian_matching(gt_boxes_in, pred_boxes_in, new_w, new_h):
    """
    Perform Hungarian matching between ground truth and predicted boxes to find the best match based on IoU scores.
    Only considers matches where IoU is greater than 0.4.

    Args:
    - gt_boxes_in (Tensor): A tensor of ground truth bounding boxes.
    - pred_boxes_in (Tensor): A tensor of predicted bounding boxes.

    Returns:
    - list of tuples: Matched pairs of ground truth and predicted boxes.
    - Tensor: IoU scores for the matched pairs.
    """
    # Check if gt_boxes is empty
    if gt_boxes_in.size(0) == 0:
        return [], None
    
    # Compute the batch IoUs
    gt_boxes = gt_boxes_in.clone().to(torch.float)
    pred_boxes = pred_boxes_in.clone().to(torch.float)

    iou_matrix = fov_iou_batch(gt_boxes, pred_boxes, new_w, new_h)

    # Convert IoUs to cost, setting high cost for IoU < 0.4
    cost_matrix = 1 - iou_matrix.detach().cpu().numpy()
    #cost_matrix[cost_matrix > 0.8] = 2  # setting cost to a high value for IoU < 0.4

    # Apply Hungarian matching
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    # Filter and Extract the matched pairs with IoU > 0.4
    matched_pairs = [(gt_indices[i], pred_indices[i]) for i in range(len(gt_indices))] #if iou_matrix[gt_indices[i], pred_indices[i]] > 0.2]

    return matched_pairs, iou_matrix


def box_center_distance(box1, box2):
    return torch.sqrt((box1[0] - box2[0])**2 + (box1[1] - box2[1])**2)

def custom_loss_function(det_preds, conf_preds, boxes, labels, class_preds, new_w, new_h):
    """
    Calculate the custom loss for an object detection model.

    This function computes a composite loss that includes confidence loss, 
    localization loss, classification loss, and a penalty for unmatched detections.
    Localization loss is applied only to detections with confidence above a 
    specified threshold, aligning this approach closer to Faster R-CNN's methodology.

    Parameters:
    - det_preds (Tensor): The predicted bounding boxes of shape (N, 4), where N is 
                          the number of detections, and each bounding box is 
                          represented as (x1, y1, x2, y2).
    - conf_preds (Tensor): The confidence scores for each predicted bounding box, 
                           of shape (N,).
    - boxes (Tensor): The ground truth bounding boxes of shape (M, 4), where M is 
                      the number of ground truth objects.
    - labels (Tensor): The ground truth labels for each object, of shape (M,).
    - class_preds (Tensor): The class predictions for each detected bounding box, 
                            of shape (N, num_classes).
    - new_w (int/float): The width scaling factor for coordinate normalization.
    - new_h (int/float): The height scaling factor for coordinate normalization.

    Returns:
    - total_loss (Tensor): The computed total loss as a scalar tensor.
    - matches (list of tuples): List of matched ground truth and prediction 
                                indices as (ground_truth_idx, prediction_idx).

    The loss computation involves the following steps:
    - Matching predicted and ground truth boxes using the Hungarian algorithm.
    - Computing the confidence loss using binary cross-entropy.
    - Computing the localization loss (1 - IoU) only for predictions with 
      confidence above a threshold (0.5 by default).
    - Computing the classification loss using cross-entropy.
    - Adding a penalty for each unmatched detection.

    Note:
    - The function assumes that the bounding box coordinates are normalized 
      using the provided scaling factors (new_w, new_h).
    - The IoU threshold for determining positive matches is set to 0.4.
    - The unmatched penalty is set to 2.5.
    - The loss components are normalized by the total number of matches and 
      unmatched detections.
    """
    
    iou_threshold = 0.4  # IoU threshold
    confidence_threshold = 0.4  # Confidence threshold for applying regression loss
    matches, _ = hungarian_matching(boxes, det_preds, new_w, new_h)

    total_loss = 0.0
    total_confidence_loss = 0.0
    total_localization_loss = 0.0
    total_classification_loss = 0.0
    unmatched_loss = 0.0

    matched_dets = set(pred_idx for _, pred_idx in matches)
    all_dets = set(range(len(det_preds)))
    unmatched_dets = all_dets - matched_dets

    for gt_idx, pred_idx in matches:
        gt_box = boxes[gt_idx]
        pred_box = det_preds[pred_idx]
        pred_confidence = conf_preds[pred_idx].view(-1)
        class_label = labels[gt_idx]
        pred_class = class_preds[pred_idx]

        iou = fov_iou(pred_box, gt_box)
        target_confidence = torch.tensor([1.0 if iou > iou_threshold else 0], dtype=torch.float, device=pred_confidence.device)
        confidence_loss = F.binary_cross_entropy(pred_confidence, target_confidence)
        total_confidence_loss += confidence_loss

        class_criterion = torch.nn.CrossEntropyLoss()
        classification_loss = class_criterion(pred_class.unsqueeze(0), class_label.unsqueeze(0))
        total_classification_loss += classification_loss

        # Apply localization loss only for confident predictions
        if pred_confidence.item() > confidence_threshold:
            localization_loss = 1 - iou
            total_localization_loss += localization_loss

    # Penalty for each unmatched detection
    unmatched_penalty = 0.5
    for det_idx in unmatched_dets:
        unmatched_confidence = conf_preds[det_idx].view(-1)
        unmatched_loss += F.binary_cross_entropy(unmatched_confidence, torch.tensor([0.0], dtype=torch.float, device=unmatched_confidence.device))

    total_loss = (total_confidence_loss + total_localization_loss + 0.1*total_classification_loss + unmatched_penalty * unmatched_loss) / (len(matches) + len(unmatched_dets)) if matches else unmatched_penalty * unmatched_loss * 5

    return total_loss, matches