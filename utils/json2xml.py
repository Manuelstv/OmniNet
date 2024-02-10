import os
import json
from PIL import Image
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

def create_voc_xml(annotations, image_filename, image_size):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "VOC2012"
    ET.SubElement(root, "filename").text = os.path.basename(image_filename)

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "The VOC2007 Database"
    ET.SubElement(source, "annotation").text = "PASCAL VOC2007"
    ET.SubElement(source, "image").text = "flickr"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image_size[0])
    ET.SubElement(size, "height").text = str(image_size[1])
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(root, "segmented").text = "0"

    for annotation in annotations:
        obj = ET.SubElement(root, "object")

        ET.SubElement(obj, "name").text = annotation[6]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        x_center, y_center, phi, theta, width, height = annotation[:6]
        #xmin = str(int(x_center - width / 2))
        ##ymin = str(int(y_center - height / 2))
        #max = str(int(x_center + width / 2))
        #ymax = str(int(y_center + height / 2))

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "x_center").text = str(int(x_center))
        ET.SubElement(bndbox, "y_center").text = str(int(y_center))
        ET.SubElement(bndbox, "phi").text = str(float(phi))
        ET.SubElement(bndbox, "theta").text = str(float(theta))
        ET.SubElement(bndbox, "width").text = str(float(width))
        ET.SubElement(bndbox, "height").text = str(float(height))

    return ET.ElementTree(root)

def process_annotations(input_dir, output_dir, image_dir):
    annotation_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    train_files, val_files = train_test_split(annotation_files, test_size=0.2, random_state=42)
    
    for split, files in zip(['train', 'val'], [train_files, val_files]):
        split_dir = os.path.join(output_dir, split)
        split_img_dir = os.path.join(split_dir, 'images')  # new directory for images
        os.makedirs(split_dir, exist_ok=True)
        os.makedirs(split_img_dir, exist_ok=True)  # ensure the directory exists
        
        for file in files:
            image_filename = os.path.join(image_dir, f"{os.path.splitext(file)[0]}.jpg")
            
            # Read image size from the image file
            with Image.open(image_filename) as img:
                image_size = img.size  # (width, height)
        
            with open(os.path.join(input_dir, file), 'r') as f:
                data = json.load(f)
                annotations = data['boxes']
            
            xml_tree = create_voc_xml(annotations, image_filename , image_size)
            xml_filename = os.path.join(split_dir, f"{os.path.splitext(file)[0]}.xml")
            xml_tree.write(xml_filename)
            
            # Copy image file to the new directory
            new_image_filename = os.path.join(split_img_dir, os.path.basename(image_filename))
            shutil.copy(image_filename, new_image_filename)

input_dir = '/home/mstveras/ssd-360/annotations'
output_dir = '/home/mstveras/ssd-360/train3/labels'
image_dir = '/home/mstveras/ssd-360/images'
process_annotations(input_dir, output_dir, image_dir)
