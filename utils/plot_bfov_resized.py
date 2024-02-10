import cv2
import numpy as np
import json
from numpy.linalg import norm
from skimage.io import imread

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
        resized_image = cv2.resize(image, (600,300))
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
    wrap_indices = np.where(u < 0)[0]
    non_wrap_indices = np.where(u >= 0)[0]

    # For u values crossing left boundary, wrap to the right
    if len(wrap_indices) > 0:
        u[wrap_indices] += w

    # First part: Within current boundary
    bfov_part1 = np.vstack((u[non_wrap_indices], v[non_wrap_indices])).T

    # Second part: Wrapped around
    bfov_part2 = np.vstack((u[wrap_indices], v[wrap_indices])).T

    # Render both parts
    image = Plotting.plotEquirectangular(image, bfov_part1, color)
    if len(wrap_indices) > 0:
        image = Plotting.plotEquirectangular(image, bfov_part2, color)

    return image

if __name__ == "__main__":
    image = imread('/home/mstveras/ssd-360/dataset/train/images/7fShr.jpg')
    h, w = image.shape[:2]
    with open('/home/mstveras/ssd-360/annotations/7fShr.json', 'r') as f:
        data = json.load(f)
    boxes = data['boxes']
    print(boxes)
    classes = data['class']
    color_map = {4: (0, 0, 255), 5: (0, 255, 0), 6: (255, 0, 0), 12: (255, 255, 0), 17: (0, 255, 255), 25: (255, 0, 255), 26: (128, 128, 0), 27: (0, 128, 128), 30: (128, 0, 128), 34: (128, 128, 128), 35: (64, 0, 0), 36: (0, 64, 0)}
    for i in range(len(boxes)):
        box = boxes[i]
        u00, v00, _, _, a_lat1, a_long1, class_name = box
        a_lat = np.radians(a_lat1)
        a_long = np.radians(a_long1)
        color = color_map.get(classes[i], (255, 255, 255))
        image = plot_bfov(image, v00, u00, a_long, a_lat,(0,255,0), 600,300)
    cv2.imwrite('/home/mstveras/final_image.png', image)
