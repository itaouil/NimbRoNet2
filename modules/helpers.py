# Imports
import sys
import cv2
import torch
import imutils
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from glob import glob
from scipy import ndimage
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
        
    for image_id, xml_file in enumerate(glob(xml_folder, recursive=True)):
        tree = ET.parse(xml_file)
        root = tree.getroot()
                
        # Change root based on xml format
        if (root.tag == 'annotations'):
            root = root.find('image')
        
        filename = root.find('filename').text
        
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # Annotated objects tag
        objects = root.findall('object')
        
        if not objects:
            xml_list.append((xml_file[:-4]+".jpg", -1, -1, -1, -1, -1, -1, -1, image_id))
        else:
            for member in root.findall('object'):
                box = member.find('bndbox')
                label = member.find('name').text

                xml_list.append((xml_file[:-4]+".jpg", 
                                 width, 
                                 height, 
                                 label, 
                                 int(box.find('xmin').text),
                                 int(box.find('ymin').text), 
                                 int(box.find('xmax').text),
                                 int(box.find('ymax').text), 
                                 image_id))
    
    print("Processed: ", image_id, " images")

    # Save as a CSV file
    column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'image_id']
    xml_df = pd.DataFrame(xml_list, columns=column_names).to_csv(output_file, index=None)

    #xml_df.to_csv(output_file, index=None)


def total_variation_channel(img:torch.tensor, channel:int)->int:
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


def total_variation(img:torch.tensor)->int:
    """
    Calculate the total variational loss.
    
    :param img: input to calculate loss for
    :param channel: channel to calculate loss on 
    :return: loss value
    """
    return (torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])))/img.shape[0]

def mse_loss_fn(y, y_hat):
    return ((y - y_hat)**2).sum()/y.shape[0]
    
def resize_image(image,h=480,w=640):
    t = transforms.Resize((h,w), interpolation=transforms.functional.InterpolationMode.NEAREST)
    #t = transforms.Resize((h,w), interpolation=PIL.Image.NEAREST)
    return np.array(t(image))

def resize_det(images,targets,h=480,w=640): # check which value is the default is for objects  # also try to test this method 
    t = transforms.Resize((h,w), interpolation=transforms.functional.InterpolationMode.NEAREST)
    
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

def get_predicted_centers(target):
    """
    Returns predicted centers for the detections
    :param probmap: the output probability map from the model
    """
    centers = {"ball": [], "robot": [], "goalpost": []}
    
    x_to_dict = {
        0: "ball",
        1: "robot",
        2: "goalpost"
    }

    # Iterate over batch
    for idx in range(len(target)):
        current_map = target[idx].detach().numpy()
        
        # Push empty list for current target
        centers["ball"].append([])
        centers["robot"].append([])
        centers["goalpost"].append([])
        
        # Iterate over batch predictions
        for x in range(3):
            # Create binary map
            binary_map = (current_map[x,:,:] > 0).astype(np.uint8)

            # Compute contours
            cnts = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            
            if len(cnts) > 0:
                for c in cnts:
                    # Compute moments and area of a contour
                    M = cv2.moments(c)
                    area = cv2.contourArea(c)
                    
                    if M["m00"] > 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        centers[x_to_dict[x]][idx].append((cY, cX, area))
            else:
                centers[x_to_dict[x]][idx].append((-1, -1, 0))
    
    return centers

def within_radius(target, prediction, radius):
    return (abs(target[1]-prediction[1]) <= radius and abs(target[0]-prediction[0]) <= radius)

def accuracy(predicted_map, target_map):
    """
    Computes accuracy between the predicted
    map and the target map
    
    :param predicted_map: predicted gaussian map
    :param target_map: target gaussian map
    """

    min_radius = 5
    min_area = min_radius**2    
    correct, total, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0
    
    target_centers = get_predicted_centers(target_map)
    predicted_centers = get_predicted_centers(predicted_map)
    
    for key in predicted_centers.keys():
        for batch in range(len(predicted_map)):
            # Get centers for given batch and key
            targets = target_centers[key][batch]
            predictions = predicted_centers[key][batch]
            
            for (target, prediction) in zip(targets, predictions):
                if (target[0]==-1 and target[1]==-1) and (prediction[2]<min_area or (prediction[0]==-1 and prediction[1]==-1)):
                    tn += 1
                elif (target[0]>=0 and target[1]>=0) and prediction[2]<min_area:
                    fn += 1
                elif (target[0]>=0 and target[1]>=0) and prediction[2]>=min_area and within_radius(target, prediction, min_radius):
                    tp += 1
                else:
                    fp += 1
    
    return tp/(tp+fp), tp/(tp+fn)
                
                
            
                
def to_device(data, device):
    """
        Move tensor(s) to chosen device
    """
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)            
            
    
    
    
            
    
    
    