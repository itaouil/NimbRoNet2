# Imports
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as ET
from torchvision import transforms

np.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt


# PARAMETERS FOR NORMALIZATION
STD_R = 0.229
STD_G = 0.224
STD_B = 0.225
MEAN_R = 0.485
MEAN_G = 0.456
MEAN_B = 0.406

MASK_MAPPING_RGB = {
    (0, 0, 0): (0,0,1),  # background
    (128, 128, 0): (0,1,0),  # field
    (0, 128, 0): (1,0,0),  # lines
    #(128, 0, 0): 1,  # ball as a field
}

MASK_MAPPING_LABEL = {
    (0,0,0): (0,0,1),  # background
    (2,3,2): (0,1,0),  # field
    (2,2,1): (1,0,0),  # lines
    (0,1,1): (0,1,0),  # ball as a field
}

def read_image(path: str) -> np.array:
    """
    Reads in an image as a numpy array.
    
    :param path: Path to the image.
    :return: Image in Numpy array format
    """
    
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def default_transforms() -> transforms.Compose:
    """
    Returns the default transformations that should be
    applied to any image passed to the model.
    
    :return: A torchvision `transforms.Compose' object containing a transforms.ToTensor object and
    transforms.Normalize.
    """

    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[MEAN_R, MEAN_G, MEAN_B],
                                                                           std=[STD_R, STD_G, STD_B])])

def reverse_normalize(image: torch.tensor) -> torch.tensor:
    """
    Reverses the normalization applied on an image.
    
    :param image: A normalized image.
    :return: The image with the normalization undone.
    """

    reverse = transforms.Normalize(mean=[-MEAN_R / STD_R, -MEAN_G / STD_G, -MEAN_B / STD_B],
                                   std=[1 / STD_R, 1 / STD_G, 1 / STD_B])
    return reverse(image)


def convert_image_to_label(image) -> np.array:
    """
    Converts the segmentation image to its pixel labeled equivalent according to the
    MASK_MAPPING parameter.
    
    :param image: RGB segmentation image
    :return: Pixel labeled image
    """        
    out = (np.zeros(image.shape))

    if (128 in np.unique(image)):
        for k in MASK_MAPPING_RGB:
            out[(image == k).all(axis=-1)] = MASK_MAPPING_RGB[k]
    else:
        for k in MASK_MAPPING_LABEL:
            out[(image == k).all(axis=-1)] = MASK_MAPPING_LABEL[k]

    return out


def convert_label_to_image(label) -> np.array:
    """
    Converts the pixel labeled image back to its RGB equivalent according to the
    MASK_MAPPING parameter
    
    :param label: Pixel labeled image
    :return: RGB segmentation image
    """
    out = (np.zeros((label.shape[0], label.shape[1], 3)))
    inverse_mask = {value: key for key, value in MASK_MAPPING_RGB.items()}

    for k in inverse_mask:
        out[(label == k).all(axis=-1)] = inverse_mask[k]

    return out


def xml_to_csv(xml_folder: str, output_file: str = 'labels.csv'):
    """
    Converts a folder of XML label files into a CSV file. Each XML file should
    correspond to an image and contain the image name, image size,
    image_id and the names and bounding boxes of the objects in the
    image, if any. Any other data is in the XML files it  willy be ignored.
    
    :param xml_folder: The path to the folder containing the XML files.
    :param output_file: Saves a CSV file containing
        the XML data in the file output_file.
    """

    xml_list = []
    image_id = 0
        
    # Loop through every XML file
    for xml_file in glob(xml_folder, recursive=True):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        extension = None
        
        # Some annotations are different
        # hence choose the root accordingly
        if (root.tag == 'annotations'):
            root = root.find('image')
            extension = root.find('filename').text[-4:]
        else:
            extension = root.find('path').text[-4:]
        
        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # Each object represents each actual image label
        for member in root.findall('object'):
            box = member.find('bndbox')
            label = member.find('name').text

            # Add image file name, image size, label, and box coordinates to CSV file
            row = (xml_file[:-4]+extension, width, height, label, int(float(box.find('xmin').text)),
                   int(float(box.find('ymin').text)), int(float(box.find('xmax').text)),
                   int(float(box.find('ymax').text)), image_id)
            xml_list.append(row)

        image_id += 1
    
    print("Processed " + str(image_id) + " images")

    # Save as a CSV file
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'image_id']
    xml_df = pd.DataFrame(xml_list, columns=column_names)

    xml_df.to_csv(output_file, index=None)
    

def resize(batch):
    """
        TO BE FIXED
    """
    """
    # Perform transformations
    width = object_entries.iloc[0, 1]
    height = object_entries.iloc[0, 2]

    # Apply the transforms manually to be able to deal with
    # transforms like Resize or RandomHorizontalFlip
    updated_transforms = []
    scale_factor = 1.0
    random_flip = 0.0
    for t in self.transform.transforms:
        # Add each transformation to our list
        updated_transforms.append(t)

        # If a resize transformation exists, scale down the coordinates
        # of the box by the same amount as the resize
        if isinstance(t, transforms.Resize):
            original_size = min(height, width)
            scale_factor = original_size / t.size

        # If a horizontal flip transformation exists, get its probability
        # so we can apply it manually to both the image and the boxes.
        elif isinstance(t, transforms.RandomHorizontalFlip):
            random_flip = t.p

    # Apply each transformation manually
    for t in updated_transforms:
        # Handle the horizontal flip case, where we need to apply
        # the transformation to both the image and the box labels
        if isinstance(t, transforms.RandomHorizontalFlip):
            # If a randomly sampled number is less the the probability of
            # the flip, then flip the image
            if random.random() < random_flip:
                image = transforms.RandomHorizontalFlip(1)(image)
                for idx, box in enumerate(targets['boxes']):
                    # Flip box's x-coordinates
                    box[0] = width - box[0]
                    box[2] = width - box[2]
                    box[[0, 2]] = box[[2, 0]]
                    targets['boxes'][idx] = box
        else:
            image = t(image)

    # Scale down box if necessary
    if scale_factor != 1.0:
        for idx, box in enumerate(targets['boxes']):
            box = (box / scale_factor).long()
            targets['boxes'][idx] = box
    """