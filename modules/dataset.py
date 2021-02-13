#TODO: Some images aren't the same size.

# Imports
import os
import math
import torch
import random
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from torchvision import transforms, utils
from scipy.stats import multivariate_normal
from torch.utils.data import Dataset, DataLoader
from helpers import reverse_normalize, read_image, default_transforms, convert_image_to_label,convert_label_to_image,resize_image


"""
For Illyas: I made the default transformations mandatory to both the datasets, and I corrected the segmentation dataset 
"""
class MyDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: Dataset, **kwargs):
        """
        Accepts a Dataset object and creates an iterable over the data
        
        :param dataset: The dataset for iteration over.
        :param kwargs: (Optional) Additional arguments to customize the
            DataLoader, such as 'batch_size' or 'shuffle'.
        """
        super().__init__(dataset, collate_fn=MyDataLoader.collate_data, **kwargs)

    # Converts a list of tuples into a tuple of lists so that
    # it can properly be fed to the model for training
    @staticmethod
    def collate_data(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)


class MyDecDataset(Dataset):
    def __init__(self, label_data: str, image_folder: str = '', h=480,w=640):
        """
        Takes as input the path to a csv file and the path to the folder where the images are located
        
        :param: label_data:
            path to a CSV file, the file should have the following columns in
            order: 'filename', 'width', 'height', 'class', 'xmin',
            'ymin', 'xmax', 'ymax' and 'image_id'
        :param: image_folder:
            path to the folder containing the images.
        :param: transform(Optional):
            a torchvision transforms object containing all transformations to be applied on all elements of the dataset
            (all box coordinates are also automatically adjusted to match the modified image-only horizontal flip
            and scaling are supported-)
        """
        # Class members
        self.width = w
        self.height = h
        self.root_dir = image_folder
        self.transform = default_transforms()
        self.labels_dataframe = pd.read_csv(label_data) # convert csv file into a pandas dataframe
        
        # Color mapping
        self.label_map = {
            "ball": 0,
            "robot": 1,
            "goalpost": 2
        }
    
    def get_probmap(self, boxes: list, labels: list, original_height: int, original_width: int) -> np.array:
        """
        Takes as input the boxes and labels
        for a given image and returns the
        respective probability map
        
        :param boxes: bouding boxes coordinates
        :param labels: labels of the boundig boxes
        :param size: probability map size
        :return: torch tensor representing a 2D image
        """
        
        # Probability map
        probmap = np.zeros((self.height, self.width, 3), dtype='float32')
        
        # Populate probability map
        for idx, coordinates in enumerate(boxes):
            # Extract coordinates (zero based)
            xmin, xmax = coordinates[0]-1, coordinates[2]-1
            ymin, ymax = coordinates[1]-1, coordinates[3]-1
                        
            # Convert coordinates from original
            # resolution to new defined resolution
            xmin = ((xmin * self.height) // original_height)
            xmax = ((xmax * self.height) // original_height)
            ymin = ((ymin * self.width) // original_width)
            ymax = ((ymax * self.width) // original_width)
                        
            # Center and radius of the box
            radius = min((xmax-xmin)/2, (ymax-ymin)/2)
            center = np.array([ymin + (ymax-ymin)/2, xmin + (xmax-xmin)/2])
                        
            # Increase radius (30%) if label is robot
            if labels[idx] == "robot":
                radius += 0.6 * radius
                        
            # Distribution
            idxs = np.meshgrid(np.arange(ymin, ymax+1), np.arange(xmin, xmax+1))
            idxs = np.array(idxs).T.reshape(-1,2)
            dist = multivariate_normal.pdf(idxs, center, [radius, radius])
            
            # Populate probability map
            probmap[ymin:ymax+1, xmin:xmax+1, self.label_map[labels[idx]]] = dist.reshape((ymax-ymin)+1, (xmax-xmin)+1)
                  
        return probmap * 100
        
    def __len__(self) -> int:
        """
        :return: the length of the dataset
        """
        return len(self.labels_dataframe['image_id'].unique().tolist())

    def __getitem__(self, idx: int) -> (torch.tensor, dict):
        """
        :param idx: index of image we want to get
        :return: tuple containing image and target dictionary
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Read in the image from the file name in the 0th column
        object_entries = self.labels_dataframe.loc[self.labels_dataframe['image_id'] == idx]
        
        img_path = os.path.join(self.root_dir, object_entries.iloc[0, 0])
        image = read_image(img_path)
        
        # Fetch labels and respective boxes
        boxes, labels = [], []
        for object_idx, row in object_entries.iterrows():

            # Read in xmin, ymin, xmax, and ymax
            box = self.labels_dataframe.iloc[object_idx, 4:8]
            boxes.append(box)

            # Read in the label
            label = self.labels_dataframe.iloc[object_idx, 3]
            labels.append(label)

        # Get relative probability map
        probmap = self.get_probmap(boxes, labels, image.shape[0], image.shape[1])
            
        boxes = torch.tensor(boxes).view(-1, 4)

        targets = {'boxes': boxes, 'labels': labels}
        
        # Perform transformations
        image = self.transform(image)
        probmap = transforms.ToTensor()(probmap)
        
        # Resize image
        resize_image(image,self.height,self.width)
        
        return image, probmap

    
class MySegDataset(Dataset):
    def __init__(self, base_dir: str, h=480,w=640):
        """
        Takes as input the path to the directory containing the images and labels
        and any transformations to be applied on the images .
        :param: base_dir:
            path to the folder containing the image folder which contains the images
            and target folder which contains the segmentation results.
        :param: transform(Optional):
            a torchvision transforms object containing all transformations to be applied on all elements of the dataset
            (all box coordinates are also automatically adjusted to match the modified image-only horizontal flip
            and scaling are supported-)
        """
        
        # Get image and labels paths for segmentation data
        self.img_paths = glob(base_dir + "/**/image/*.jpg", recursive=True)
        self.lbl_paths = glob(base_dir + "/**/target/*.png", recursive=True)
        
        # Sort lists to have a 1:1 mapping
        self.img_paths.sort()
        self.lbl_paths.sort()
                
        # Make sure we have the same size for both
        assert len(self.img_paths) == len(self.lbl_paths)

        # Defaults transformations
        self.transform = default_transforms()
        
        # Orignal size
        self.h = h
        self.w = w

    def __len__(self) -> int:
        """
        :return: the length of the dataset
        """
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        :param idx: index of image we want to get
        :return: tuple containing image and segmentation label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image
        image = read_image(str(self.img_paths[idx]))

        # Get label
        label = read_image(str(self.lbl_paths[idx]))
        
        # Resize images and labels
        image = transforms.ToPILImage()(image)
        label = transforms.ToPILImage()(label)
        image = resize_image(image,self.h,self.w)
        label = resize_image(label,self.h,self.w)
        
        # Convert image to one channel labels
        label = convert_image_to_label(label)
        
        # Perform transformations
        image = self.transform(image)
        label = transforms.ToTensor()(label)
                
        return image, label