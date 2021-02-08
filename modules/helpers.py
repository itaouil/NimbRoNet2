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
    (0, 0, 0): 0,  # background
    (128, 128, 0): 1,  # field
    (0, 128, 0): 2,  # lines
    #(128, 0, 0): 1,  # ball as a field
}
MASK_MAPPING_LABEL = {
    (0,0,0): 0,  # background
    (1,1,1): 1,  # field
    (2,2,2): 2,  # lines
    (3,3,3): 1,  # ball as a field
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
    
    
    out = (np.zeros(image.shape[:2]))
    if (128 in np.unique(image)):
        for k in MASK_MAPPING_RGB:
            out[(image == k).all(axis=2)] = MASK_MAPPING_RGB[k]
    else:
        for k in MASK_MAPPING_LABEL:
            out[(image == k).all(axis=2)] = MASK_MAPPING_LABEL[k]

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
        out[(label == k)] = inverse_mask[k]

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
    
def total_variation_loss(img:torch.tensor, channel:int)->int:
    """
    Calculate the total variational loss for an image across a single channel
    :param img: input to calculate loss for
    :param channel: channel to calculate loss on 
    :return: loss value
    """
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,channel,1:,:]-img[:,channel,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,channel,:,1:]-img[:,channel,:,:-1], 2).sum()
    return (tv_h+tv_w)/(bs_img*h_img*w_img)
    
def resize_seg(images,labels,h=480,w=640):
    t = transforms.Resize((h,w))
    return t(images),t(labels)
    

def resize_det(images,targets,h=480,w=640): # check which value is the default is for objects  # also try to test this method 
    t = transforms.Resize((h,w))
    
    original_size = min(images[2], images[3])
    scale_factor = original_size / transform.size
    
    for idx, box in enumerate(targets['boxes']):
            box = (box / scale_factor).long()
            targets['boxes'][idx] = box
    
    return t(images),targets
    
def to_device(data:torch.tensor, device:str):
    """
        Move tensor(s) to chosen device
        :param data: input tensor
        :param device: device to move tensor to
        
    """
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)