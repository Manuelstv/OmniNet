import cv2
import numpy as np
import json
from numpy.linalg import norm
from skimage.io import imread
import math

def equirectangular_to_pixel(eqr_width, eqr_height, center_latitude, center_longitude , equirectangular_latitude, equirectangular_longitude):
    """
    Converts equirectangular coordinates to pixel coordinates in a rectangular image.

    Parameters:
    - eqr_width (int): Width of the equirectangular image in pixels.
    - eqr_height (int): Height of the equirectangular image in pixels.
    - center_latitude (float): Latitude of the equirectangular image's center (in degrees).
    - center_longitude (float): Longitude of the equirectangular image's center (in degrees).
    - equirectangular_latitude (float): Latitude in equirectangular projection to convert (in degrees).
    - equirectangular_longitude (float): Longitude in equirectangular projection to convert (in degrees).

    Returns:
    - pixel_x (int): Pixel X-coordinate in the rectangular image.
    - pixel_y (int): Pixel Y-coordinate in the rectangular image.
    """
    # Calculate angular offsets from the center
    latitude_offset = equirectangular_latitude - center_latitude
    longitude_offset = equirectangular_longitude - center_longitude

    # Calculate pixel offsets per degree
    pixels_per_degree_x = eqr_width / 360.0
    pixels_per_degree_y = eqr_height / 180.0

    # Calculate pixel coordinates in the rectangular image
    pixel_x = int((longitude_offset * pixels_per_degree_x) + (eqr_width / 2))
    pixel_y = int((-latitude_offset * pixels_per_degree_y) + (eqr_height / 2))

    return pixel_x, pixel_y


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
        resized_image = image
        kernel = kernel.astype(np.int32)
        hull = cv2.convexHull(kernel)
        cv2.polylines(resized_image,
                      [hull],
                      isClosed=True,
                      color=color,
                      thickness=2)
        return resized_image


def plot_circles(img, arr, color):
    for i in arr:
        img = cv2.circle(img, i, 10, color, -1)
    return img


def plot_bfov(image, v00, u00, a_lat, a_long, color, h, w):
    t = int(w//2 - u00)
    u00 += t
    image = np.roll(image, t, axis=1)

    phi00 = (u00 - w / 2.) * ((2. * np.pi) / w)
    theta00 = -(v00 - h / 2.) * (np.pi / h)
    r = 10
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
    kernel = np.stack((u, v), axis=-1).astype(np.int32)
    image = plot_circles(image, kernel, color)

    image = Plotting.plotEquirectangular(image, kernel, color)
    image = np.roll(image, w - t, axis=1)

    return image

if __name__ == "__main__":
    image = imread('image.png')
    h, w = image.shape[:2]
    color_map = {1: (47,  52,  227),
                 2: (63,  153, 246),
                 3: (74,  237, 255),
                 4: (114, 193, 56),
                 5: (181, 192, 77),
                 6: (220, 144, 51),
                 7: (205, 116, 101),
                 8: (226, 97,  149),
                 9: (155, 109, 246)}

    v00, u00 = 117,426

    def deg_to_rad(degrees):
        return [math.radians(degree) for degree in degrees]
    #a_lat = np.pi / 4
    #a_long = np.pi/6

    #(0,180) and (0,360)
    print(equirectangular_to_pixel(eqr_width = w, eqr_height = h , center_latitude =0, center_longitude =0, equirectangular_latitude = 72, equirectangular_longitude =169))
    
    a_lat, a_long = deg_to_rad([45,30])
    color = (255, 0, 0)
    image = plot_bfov(image, v00, u00, a_lat, a_long, color, h, w)

    a_lat, a_long = deg_to_rad([65,51])
    color = (0, 255, 0)

    v00, u00 = 1861,96
    image = plot_bfov(image, v00, u00, a_lat, a_long, color, h, w)

    image = cv2.circle(image, (u00, v00), 5, (255,0,0), -1)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    cv2.imwrite('bfov_transl.png', image)