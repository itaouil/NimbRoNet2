import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from utils import reverse_normalize, convert_label_to_image


def show_labeled_image(image: torch.tensor or np.array, boxes: torch.tensor, labels: list = None):
    """
    Show the image along with the specified boxes around detected objects.
    Also displays each box's label if a list of labels is provided.
    :param image: The image to plot. If the image is a torch.Tensor object,
         it will automatically be reverse-normalized
        and converted to a PIL image for plotting.
    :param boxes: A torch tensor of size (N, 4) where N is the number
        of boxes to plot.
    :param labels: (Optional) A list of size N giving the labels of
            each box.
    """
    fig, ax = plt.subplots(1)

    # If the image is already a tensor, convert it back to a PILImage and reverse normalize it
    if isinstance(image, torch.Tensor):
        image = reverse_normalize(image)
        image = transforms.ToPILImage()(image)

    ax.imshow(image)

    # Show a single box or multiple if provided
    if boxes.ndim == 1:
        boxes = boxes.view(1, 4)

    # Convert labels to list
    if labels is not None and not (isinstance(labels, list) or isinstance(labels, tuple)):
        labels = [labels]

    # Plot each box
    for i in range(boxes.shape[0]):
        box = boxes[i]
        width, height = (box[2] - box[0]).item(), (box[3] - box[1]).item()
        initial_pos = (box[0].item(), box[1].item())
        rect = patches.Rectangle(initial_pos,  width, height, linewidth=1,
                                 edgecolor='r', facecolor='none')
        if labels:
            ax.text(box[0] + 5, box[1] - 5, '{}'.format(labels[i]), color='red')

        ax.add_patch(rect)

    plt.show()


def show_image_and_seg(image: torch.tensor or np.array, labels: np.array):
    """
    Show the image along with its segmentation.
    :param image: The image to plot. If the image is a torch.Tensor object,
         it will automatically be reverse-normalized
        and converted to a PIL image for plotting.
    :param labels: The segmentation of the image.
    """
    fig, ax = plt.subplots(1, 2)

    # If the image is already a tensor, convert it back to a PILImage and reverse normalize it
    if isinstance(image, torch.Tensor):
        image = reverse_normalize(image)
        image = transforms.ToPILImage()(image)

    ax[0].imshow(image)
    ax[1].imshow(convert_label_to_image(labels).astype(np.uint8))

    plt.show()
