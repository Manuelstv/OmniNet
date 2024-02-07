import numpy as np
import cv2
from numpy.linalg import norm
import math
from skimage.io import imread
from numpy.linalg import norm


def equirectangular_to_pixel(eqr_width, eqr_height, center_latitude, center_longitude, equirectangular_latitude, equirectangular_longitude):
    """
    Converts equirectangular coordinates to pixel coordinates.

    Parameters:
    eqr_width (int): Width of the equirectangular image.
    eqr_height (int): Height of the equirectangular image.
    center_latitude (float): Center latitude of the image.
    center_longitude (float): Center longitude of the image.
    equirectangular_latitude (float): Latitude to convert.
    equirectangular_longitude (float): Longitude to convert.

    Returns:
    Tuple[int, int]: Pixel coordinates (x, y) in the image.
    """
    latitude_offset = equirectangular_latitude - center_latitude
    longitude_offset = equirectangular_longitude - center_longitude

    pixels_per_degree_x = eqr_width / 360.0
    pixels_per_degree_y = eqr_height / 180.0

    pixel_x = int((longitude_offset * pixels_per_degree_x) + (eqr_width / 2))
    pixel_y = int((-latitude_offset * pixels_per_degree_y) + (eqr_height / 2))

    return pixel_x, pixel_y

class Rotation:
    @staticmethod
    def Rx(alpha):
        """ Rotation matrix for rotation around the x-axis. """
        return np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])

    @staticmethod
    def Ry(beta):
        """ Rotation matrix for rotation around the y-axis. """
        return np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])

    @staticmethod
    def Rz(gamma):
        """ Rotation matrix for rotation around the z-axis. """
        return np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

class Plotting:
    @staticmethod
    def plotEquirectangular(image, kernel, color):
        """
        Draws a polyline on an image using a convex hull of provided points.

        Parameters:
        image (ndarray): The image on which to draw.
        kernel (ndarray): Array of points to create the polyline.
        color (tuple): Color of the polyline.

        Returns:
        ndarray: The image with the polyline drawn.
        """
        resized_image = image
        hull = cv2.convexHull(kernel)
        cv2.polylines(resized_image, [hull], isClosed=True, color=color, thickness=2)
        return resized_image

def plot_circles(img, arr, color, transparency):
    """
    Draws transparent circles on an image at specified coordinates.

    Parameters:
    img (ndarray): The image on which to draw.
    arr (list): List of center coordinates for the circles.
    color (tuple): Color of the circles.
    transparency (float): Transparency of the circles.

    Returns:
    ndarray: The image with circles drawn.
    """
    overlay = img.copy()
    for point in arr:
        cv2.circle(overlay, point, 10, color, -1)
    
    cv2.addWeighted(overlay, transparency, img, 1 - transparency, 0, img)
    return img

def plot_bfov(image, v00, u00, a_lat, a_long, color, h, w):
    """
    Plots a bounding field of view on an equirectangular image.

    Parameters:
    image (ndarray): The equirectangular image.
    v00, u00 (int): Pixel coordinates of the center of the field of view.
    a_lat, a_long (float): Angular size of the field of view in latitude and longitude.
    color (tuple): Color of the plot.
    h, w (int): Height and width of the image.

    Returns:
    ndarray: The image with the field of view plotted.
    """
    t = int(w//2 - u00)
    u00 += t
    image = np.roll(image, t, axis=1)

    phi00 = (u00 - w / 2.) * ((2. * np.pi) / w)
    theta00 = -(v00 - h / 2.) * (np.pi / h)
    r = 30
    d_lat = r / (2 * np.tan(a_lat / 2))
    d_long = r / (2 * np.tan(a_long / 2))
    p = [np.array([i * d_lat / d_long, j, d_lat]) for i in range(-(r - 1) // 2, (r + 1) // 2) for j in range(-(r - 1) // 2, (r + 1) // 2)]
    R = np.dot(Rotation.Ry(phi00), Rotation.Rx(theta00))
    p = [np.dot(R, (point / norm(point))) for point in p]
    phi = [np.arctan2(point[0], point[2]) for point in p]
    theta = [np.arcsin(point[1]) for point in p]
    u = [(angle / (2 * np.pi) + 0.5) * w for angle in phi]
    v = [h - (-angle / np.pi + 0.5) * h for angle in theta]

    u = np.nan_to_num(u, nan=0)
    v = np.nan_to_num(v, nan=0)

    kernel = np.array([u, v], dtype=np.int32).T
    image = plot_circles(image, kernel, color, 0.25)

    image = Plotting.plotEquirectangular(image, kernel, color)
    image = np.roll(image, w - t, axis=1)

    return image

def process_and_save_image(images, matches, gt_boxes, confidences, det_preds, class_preds, labels, threshold, color_gt, save_path):
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