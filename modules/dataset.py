# Imports
import os
import torch
import random
import numpy as np
import pandas as pd
from glob import glob

from pathlib import Path
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from utils import reverse_normalize, read_image, default_transforms, convert_image_to_label

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
    def __init__(self, label_data: str, image_folder: str = '', transform: transforms = None):
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
        self.root_dir = image_folder

        # convert csv file into a pandas dataframe
        self.labels_dataframe = pd.read_csv(label_data)

        if transform is None:
            self.transform = default_transforms()
        else:
            self.transform = transform

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

        boxes = []
        labels = []
        for object_idx, row in object_entries.iterrows():

            # Read in xmin, ymin, xmax, and ymax
            box = self.labels_dataframe.iloc[object_idx, 4:8]
            boxes.append(box)

            # Read in the label
            label = self.labels_dataframe.iloc[object_idx, 3]
            labels.append(label)

        boxes = torch.tensor(boxes).view(-1, 4)

        targets = {'boxes': boxes, 'labels': labels}

        # Perform transformations
        if self.transform:
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

        return image, targets

    
class MySegDataset(Dataset):
    def __init__(self, base_dir: str, transform: transforms = None):
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
        
        self.img_paths = glob(base_dir + "/**/image/*.jpg", recursive=True)
        self.lbl_paths = glob(base_dir + "/**/target/*.png", recursive=True)
                
        assert len(self.img_paths) == len(self.lbl_paths)

        if transform is None:
            self.transform = default_transforms()
        else:
            self.transform = transform

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
        img_path = self.img_paths[idx]
        image = read_image(str(img_path))

        # Get label
        label_path = self.lbl_paths[idx]
        label = read_image(str(label_path))

        # Perform transformations
        image = self.transform(image)
        label = self.transform(label)

        # Undo normalization done to label
        label = reverse_normalize(label)
        label = np.array(transforms.ToPILImage()(label))

        # Get the label for each pixel
        label = convert_image_to_label(label)

        return image, label