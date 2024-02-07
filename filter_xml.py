import os
import shutil  # Import shutil for file copying
import xml.etree.ElementTree as ET

# Directory paths
input_dir = '/home/mstveras/OmniNet/dataset/val/labels/'  # Directory with XML files
image_dir = '/home/mstveras/OmniNet/dataset/val/images/'  # Directory with image files
output_dir_labels = '/home/mstveras/OmniNet/dataset/val/labels2/'  # Output directory for filtered XMLs
output_dir_images = '/home/mstveras/OmniNet/dataset/val/images2'  # Output directory for corresponding images

# Create the output directories if they do not exist
os.makedirs(output_dir_labels, exist_ok=True)
os.makedirs(output_dir_images, exist_ok=True)

def filter_xml_and_copy_images(input_file_path, output_label_path, output_image_dir, image_dir, filter_names):
    # Parse the input XML file
    tree = ET.parse(input_file_path)
    root = tree.getroot()

    # Create a new XML element to hold the filtered objects
    annotation = ET.Element('annotation')
    object_found = False

    for elem in root:
        if elem.tag != 'object':
            annotation.append(elem)

    for obj in root.findall('object'):
        object_name = obj.find('name').text
        if object_name in filter_names:
            annotation.append(obj)
            object_found = True

    if object_found:
        # Write the filtered XML to the new file
        new_tree = ET.ElementTree(annotation)
        new_tree.write(output_label_path, encoding='utf-8', xml_declaration=True)

        # Copy the corresponding image file
        base_filename = os.path.splitext(os.path.basename(input_file_path))[0]
        for ext in ['.jpg', '.jpeg', '.png']:  # Add or remove extensions as needed
            image_path = os.path.join(image_dir, base_filename + ext)
            if os.path.exists(image_path):
                shutil.copy(image_path, output_image_dir)
                break  # Stop searching after the first match to avoid copying multiple times
    else:
        print(f"No specified objects found in: {os.path.basename(input_file_path)}")

def filter_xml_files_and_copy_images(input_dir, output_dir_labels, output_dir_images, image_dir, filter_names):
    for filename in os.listdir(input_dir):
        if filename.endswith('.xml'):
            input_file_path = os.path.join(input_dir, filename)
            output_label_path = os.path.join(output_dir_labels, filename)
            filter_xml_and_copy_images(input_file_path, output_label_path, output_dir_images, image_dir, filter_names)

# Define the names to keep
names_to_keep = ['person', 'chair', 'light', 'door', 'picture']

# Apply the filtering and copying to all XML files and their corresponding images in the directory
filter_xml_files_and_copy_images(input_dir, output_dir_labels, output_dir_images, image_dir, names_to_keep)
