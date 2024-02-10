def pixel_to_spherical(x, y, image_width, image_height):
    # Normalize pixel coordinates to [-1, 1]
    x_normalized = (x / (image_width / 2)) - 1
    y_normalized = (y / (image_height / 2)) - 1

    # Convert normalized coordinates to spherical coordinates
    theta = x_normalized * 360 / 2
    phi = y_normalized * 180 / 2

    return theta, phi

# Assume a bounding box in pixel coordinates
#bbox_pixel_coords = [x_c, y_c, alpha, beta]
bbox_pixel_coords = [0, 300,4,4]

def multiply_columns(tensor):
    # Check if the input tensor has 4 columns
    if tensor.shape[1] != 5:
        raise ValueError("Input tensor must have 4 columns.")

    # Create a copy of the input tensor to avoid modifying the original
    result_tensor = tensor.clone()

    # Perform column-wise multiplications
    result_tensor[:, 0] *= 600
    result_tensor[:, 1] *= 300
    result_tensor[:, 2] *= 90
    result_tensor[:, 3] *= 90
    result_tensor[:, 4] = 0

    return result_tensor


gt1 = torch.tensor(np.array([
    [0.1250, 0.5560, 21.0000/90 , 93.0000 / 90, 0],
    [0.4350, 0.5460, 13.0000/90 , 69.0000 / 90, 0],
    [0.7300, 0.4810, 69.0000/90 ,65.0000 / 90 , 0],
    [0.4820, 0.4720,  5.0000/90 , 29.0000 / 90 , 0],
    [0.2960 , 0.5590, 93.0000/90, 109.0000 / 90, 0]
]))
# Image dimensions and camera FoV
image_width = 600  # Example width
image_height = 300  # Example height
HFOV = 360  # Example HFOV for a 360° camera
VFOV = 180  # Example VFOV for a 360° camera

# Calculate center coordinates in spherical format
theta_center, phi_center = pixel_to_spherical(
    bbox_pixel_coords[0],
    bbox_pixel_coords[1],
    image_width, image_height)

# The final FoV-BB format
print((theta_center, phi_center))
