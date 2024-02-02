import numpy as np
import cv2
from numpy.linalg import norm


class Rotation:
    @staticmethod
    def Rx(alpha):
        return np.asarray([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    @staticmethod
    def Ry(beta):
        return np.asarray([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    @staticmethod
    def Rz(gamma):
        return np.asarray([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

class Plotting:
    @staticmethod
    def plotEquirectangular(image, kernel, color):
        resized_image = np.ascontiguousarray(image, dtype=np.uint8)
        kernel = kernel.astype(np.int32)
        hull = cv2.convexHull(kernel)
        cv2.polylines(resized_image, [hull], isClosed=True, color=color, thickness=2)
        return resized_image


def plot_bfov(image, v00, u00, a_lat, a_long, color, h, w):
    phi00 = (u00 - w / 2.) * ((2. * np.pi) / w)
    theta00 = -(v00 - h / 2.) * (np.pi / h)
    r = 100
    d_lat = r / (2 * np.tan(a_lat / 2))
    d_long = r / (2 * np.tan(a_long / 2))
    p = []
    for i in range(-(r - 1) // 2, (r + 1) // 2):
        for j in range(-(r - 1) // 2, (r + 1) // 2):
            p += [np.asarray([i * d_lat / d_long, j, d_lat])]
    R = np.dot(Rotation.Ry(phi00), Rotation.Rx(theta00))
    p = np.asarray([np.dot(R, (p[ij] / norm(p[ij]))) for ij in range(r * r)])
    phi = np.asarray([np.arctan2(p[ij][0], p[ij][2]) for ij in range(r * r)])
    theta = np.asarray([np.arcsin(p[ij][1]) for ij in range(r * r)])
    u = (phi / (2 * np.pi) + 1. / 2.) * w
    v = h - (-theta / np.pi + 1. / 2.) * h
    return Plotting.plotEquirectangular(image, np.vstack((u, v)).T, color)

def process_and_save_image2(images, matches, gt_boxes, confidences, det_preds, threshold, color_gt, save_path):
    """
    Process an image, plot ground truth and predictions (above a confidence threshold), and save the image.
    
    Args:
    - images: The image as a tensor.
    - gt_boxes: Ground truth bounding boxes.
    - det_preds: Detection predictions including bounding boxes.
    - confidences: Confidence scores for each prediction.
    - threshold: Confidence threshold to decide which predictions to plot.
    - color_gt: Color for the ground truth boxes.
    - color_pred: Color for the predicted boxes.
    - save_path: Path to save the processed image.
    """
    
    # Process the image for visualization
    images = images.mul(255).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()

    # Plot ground truth boxes
    for box in gt_boxes:
        u00, v00, a_lat1, a_long1 = (box[0]+1)*(600/2), (box[1]+1)*(300/2), 90*box[2], 90*box[3]
        a_long = np.radians(a_long1)
        a_lat = np.radians(a_lat1)
        images = plot_bfov(images, v00, u00, a_long, a_lat, color_gt, 300, 600)

    # Iterate through predictions and their associated confidences
    for pred, conf in zip(det_preds, confidences):
        conf = conf.item()  # Assuming conf is a tensor with a single value
        box = pred[:4]
        u00, v00, a_lat1, a_long1 = (box[0]+1)*(600/2), (box[1]+1)*(300/2), 90*box[2], 90*box[3]
        a_long = np.radians(a_long1)
        a_lat = np.radians(a_lat1)
            
        # Annotate the image with the confidence score
        label = f'{conf:.2f}'  # Format the confidence to 2 decimal places
        # Position for the text, you might need to adjust depending on the image
        label_position = (int(u00), int(v00) - 10)
        
        if conf>0.6:
            color_pred = (255,0,0)
        else:
            color_pred = (0,0,255)
        
        images = plot_bfov(images, v00, u00, a_long, a_lat, color_pred, 300, 600)
        cv2.putText(images, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_pred, 2)

    # Save the image with plotted boxes
    cv2.imwrite(save_path, images)


def process_and_save_image(images, matches, gt_boxes, det_preds, threshold, color_gt, color_pred, save_path):
    """
    Process an image, plot ground truth and predictions (above a confidence threshold), and save the image.
    
    Args:
    - images: The image as a tensor.
    - gt_boxes: Ground truth bounding boxes.
    - det_preds: Detection predictions including bounding boxes and confidence scores.
    - threshold: Confidence threshold to decide which predictions to plot.
    - color_gt: Color for the ground truth boxes.
    - color_pred: Color for the predicted boxes.
    - save_path: Path to save the processed image.
    """
    
    # Process the image for visualization
    images = images.mul(255).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()

    # Plot ground truth boxes
    for box in gt_boxes:
        u00, v00, a_lat1, a_long1 = (box[0]+1)*(600/2), (box[1]+1)*(300/2), 90*box[2], 90*box[3]
        a_long = np.radians(a_long1)
        a_lat = np.radians(a_lat1)
        images = plot_bfov(images, v00, u00, a_long, a_lat, color_gt, 300, 600)

    # Plot predictions if their confidence is greater than the threshold
    for pred in det_preds:
        #conf = torch.sigmoid(pred[4])
        #if conf > threshold:
        #    print(conf)
            box = pred[:4]
            u00, v00, a_lat1, a_long1 = (box[0]+1)*(600/2), (box[1]+1)*(300/2), 90*box[2], 90*box[3]
            a_long = np.radians(a_long1)
            a_lat = np.radians(a_lat1)
            images = plot_bfov(images, v00, u00, a_long, a_lat, color_pred, 300, 600)

    # Save the image with plotted boxes
    cv2.imwrite(save_path, images)


def process_and_save_image_planar(images, boxes, color, save_path):
    """
    Process an image from a list, draw bounding boxes on it, and save the image.

    Args:
    - images (list of np.array): A list of images, each as a NumPy array.
    - boxes (list of tuples): A list of bounding boxes, each box defined as a tuple (x_min, y_min, x_max, y_max).
    - color (tuple): Color of the bounding box in BGR (Blue, Green, Red) format.
    - save_path (str): The file path where the processed image will be saved.
    """
    # Check if images list is not empty
    #if not images:
    #    raise ValueError("The images list is empty.")

    # Process each image

    new_w, new_h = 600, 300

    #print(type(images))

    #images = np.ascontiguousarray(images, dtype = np.uint8)
    #images2 = images.copy()

    for box in boxes:
        x_min, y_min, x_max, y_max = int(box[0]*new_w), int(box[1]*new_h), int(box[2]*new_w), int(box[3]*new_h)
        #images = np.ascontiguousarray(images, dtype = np.uint8)
        cv2.rectangle(images, (x_min, y_min), (x_max, y_max), color, 2)

    # Save the processed image
    cv2.imwrite(save_path, images)
