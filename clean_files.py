import os

# Define the paths to your directories
xml_dir = '/home/mstveras/OmniNet/dataset/val/labels'
image_dir = '/home/mstveras/OmniNet/dataset/val/images'

# List all XML files in the xml_dir
xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

# Define possible image extensions
image_extensions = ['.jpg', '.jpeg', '.png']

# Go through each XML file to check if a corresponding image exists
for xml_file in xml_files:
    # Extract the base filename without extension
    base_name = os.path.splitext(xml_file)[0]
    
    # Assume no corresponding image initially
    corresponding_image_found = False
    
    # Check for each possible image extension
    for ext in image_extensions:
        image_file = base_name + ext
        if image_file in os.listdir(image_dir):
            corresponding_image_found = True
            break  # Stop searching if we found the corresponding image
    
    # If no corresponding image is found, remove the XML file
    if not corresponding_image_found:
        xml_file_path = os.path.join(xml_dir, xml_file)
        print(f"Removing XML file: {xml_file_path}")  # Optional: print statement for tracking
        os.remove(xml_file_path)

print("Cleanup complete.")
