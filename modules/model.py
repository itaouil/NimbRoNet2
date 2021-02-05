import torch
import torchvision.models as models


# For Illyas: I created a class for the location dependant convolution. For the issue that we we want to use the same model for different image sizes. So I thought I would just set the model to work with the max size, and then take the input size as a parameter in the forward method. Then depending on the image size we would take only the part of the bias parameters we need. (I will ask Hafez if this is the correct method to deal with it or not). He Said we should resize it ? but I don't know how to do that 
# I adjusted the model to use the new convolution layer
class LocationAware1X1Conv2d(torch.nn.Conv2d):
    """
    Location-Dependent convolutional layer in accordance to
    (Azizi, N., Farazi, H., & Behnke, S. (2018, October).
    Location dependency in video prediction. In International Conference on Artificial Neural Networks (pp. 630-638). Springer, Cham.)
    
    """
    def __init__(self,w,h,in_channels, out_channels, bias=True):
        super().__init__(in_channels, out_channels, kernel_size =1, bias=bias)
        self.locationBias=torch.nn.Parameter(torch.zeros(h,w,1))
    
    def forward(self,inputs,w,h):    
        b=self.locationBias
        convRes=super().forward(inputs)
        
        return convRes+b[0:h,0:w,0]

    
class Res18Encoder(torch.nn.Module):
    def __init__(self):
        """
        Initialize the encoder used in model. A pre-trained ResNet-18 is chosen as the encoder,
        but with the Global Average Pooling (GAP) and the fully connected layers removed.
        """
        super().__init__()
        resnet18 = list(models.resnet18(pretrained=True).children())

        self.conv1 = resnet18[0]
        self.bn1 = resnet18[1]
        self.relu = resnet18[2]
        self.maxpool = resnet18[3]
        self.layer1 = resnet18[4]
        self.layer2 = resnet18[5]
        self.layer3 = resnet18[6]
        self.layer4 = resnet18[7]

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Definition of the forward function of the model (called when a parameter is passed
        as through ex: model(x))
        :param x: Input Image
        :return: Four outputs of the model at different stages
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4


class Decoder(torch.nn.Module):
    def __init__(self,w:int,h:int):
        """
        Initialize the decoder used in model.
        Transpose convolutional layers are used for up-sampling,
        and location-dependent convolutional layers are used in the
        output heads with the learnable bias shared between them.
        :param w: The max width of the output image 
        :param h: The max height of the output image
        """
        super().__init__()
        self.relu = torch.nn.functional.relu
        self.convTrans1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.conv1 = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(512)
        self.convTrans2 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.convTrans3 = torch.nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(256)
        
        self.convD = LocationAware1X1Conv2d(w,h,256, 3)
        self.convS = LocationAware1X1Conv2d(w,h,256, 1)
        
        # share learnable bias between both heads by remove the locationBias from the segmentation head
        # and override it with the locationBias from the detection head
        del self.convS.locationBias
        self.convS.locationBias = self.convD.locationBias

    def forward(self, w:int, h:int, x4: torch.tensor, x3: torch.tensor, x2: torch.tensor, x1: torch.tensor, head: str) -> torch.tensor:
        """
        Definition of the forward function of the model (called when a parameter is passed
        as through ex: model(x))
        :param w: Final output of the encoder
        :param h: Output from layer 3 in the encoder
        :param x4: Final output of the encoder
        :param x3: Output from layer 3 in the encoder
        :param x2: Output from layer 2 in the encoder
        :param x1: Output from layer 1 in the encoder
        :param head: Determines which head will output the results
        :return: Either detection or segmentation result
        """
        x = self.relu(x4)
        x = self.convTrans1(x)
        x = torch.cat((x, self.conv1(x3)), 1)
        x = self.bn1(self.relu(x))
        x = self.convTrans2(x)
        x = torch.cat((x, self.conv2(x2)), 1)
        x = self.bn2(self.relu(x))
        x = self.convTrans3(x)
        x = torch.cat((x, self.conv3(x1)), 1)
        x = self.bn3(self.relu(x))
        
        if head == "segmentation":
            xs = self.convS(x,w,h)
            return xs
        elif head == "detection":
            xd = self.convD(x,w,h)
            return xd
        else:
            raise Exception("invalid head")


class Model(torch.nn.Module):
    def __init__(self,w:int, h:int):
        """
        Initialize the decoder used in model.
        Transpose-convolutional layers are used for up-sampling the representations
        :param w: Width of input image 
        :param h: Height of input image
        """
        super().__init__()
        self.encoder = Res18Encoder()
        self.decoder = Decoder(int(w/4), int(h/4))

    def forward(self, x: torch.tensor, head: str = "segmentation") -> torch.tensor:
        """
        Definition of the forward function of the model (called when a parameter is passed
        as through ex: model(x))
        :param head:
        :param x: Four Inputs from the Decoder model
        :param head: Determines which head will output the results
        :return: Either detection or segmentation result
        """
        w = int(x.shape[3]/4)
        h = int(x.shape[2]/4)
        x1, x2, x3, x4 = self.encoder(x)
        
        x = self.decoder(w,h,x4, x3, x2, x1, head)

        return x

