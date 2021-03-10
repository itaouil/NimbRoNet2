import torch
import numpy as np
import torchvision.models as models
from helpers import total_variation,to_device,accuracy,mse_loss_fn


class LocationAware1X1Conv2d(torch.nn.Conv2d):
    """
    Location-Dependent convolutional layer in accordance to
    (Azizi, N., Farazi, H., & Behnke, S. (2018, October).
    Location dependency in video prediction. In International Conference on Artificial Neural Networks (pp. 630-638). Springer, Cham.)
    """
    def __init__(self,w,h,in_channels, out_channels, bias=True):
        super().__init__(in_channels, out_channels, kernel_size =1, bias=bias)
        self.locationBias=torch.nn.Parameter(torch.rand((1, 3, h, w), requires_grad=True))
    
    def forward(self,inputs,w,h):
        # Upsample location bias to match image size
        if self.locationBias[0].shape != (3,h,w):
            upsampled_bias = torch.nn.functional.interpolate(self.locationBias, size=((h,w)), mode='nearest')
            self.locationBias = torch.nn.Parameter(upsampled_bias, requires_grad=True)

        # Perform convolution
        convRes=super().forward(inputs)
        
        # Add location bias
        return convRes + self.locationBias

    
class Res18Encoder(torch.nn.Module):
    def __init__(self):
        """
        Initialize the encoder used in model. A pre-trained ResNet-18 is chosen as the encoder,
        but with the Global Average Pooling (GAP) and the fully connected layers removed.
        """
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        
        resnet_children = list(self.resnet18.children())

        self.conv1 = resnet_children[0]
        self.bn1 = resnet_children[1]
        self.relu = resnet_children[2]
        self.maxpool = resnet_children[3]
        self.layer1 = resnet_children[4]
        self.layer2 = resnet_children[5]
        self.layer3 = resnet_children[6]
        self.layer4 = resnet_children[7]
    
    def freeze(self):
        for child in self.resnet18.children():
            for param in child.parameters():
                param.requires_grad = False
    
    def unfreeze(self):
        for child in self.resnet18.children():
            for param in child.parameters():
                param.requires_grad = True

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
        
        Defined layers are in order of usage.
        
        :param w: The max width of the output image 
        :param h: The max height of the output image
        """
        super().__init__()
        
        # ReLu
        self.relu = torch.nn.functional.relu
        
        # Transposed convolution 1
        self.convTrans1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        
        # Convolution 1
        self.conv1 = torch.nn.Conv2d(256, 256, kernel_size=1, stride=1)
        
        # Batch Norm 1
        self.bn1 = torch.nn.BatchNorm2d(512)
        
        # Transposed convolution 1
        self.convTrans2 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        
        # Convolution 2
        self.conv2 = torch.nn.Conv2d(128, 256, kernel_size=1, stride=1)
        
        # Batch Norm 2
        self.bn2 = torch.nn.BatchNorm2d(512)
        
        # Transposed convolution 3
        self.convTrans3 = torch.nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2, padding=0)
        
        # Convolution 3
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=1, stride=1)
        
        # Batch Norm 3
        self.bn3 = torch.nn.BatchNorm2d(256)
        
        self.convD = LocationAware1X1Conv2d(w,h,256, 3)
        self.convS = LocationAware1X1Conv2d(w,h,256, 3)
        
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
    def __init__(self,device,w:int,h:int):
        """
        Initialize the decoder used in model.
        Transpose-convolutional layers are used for up-sampling the representations
        :param w: Width of input image 
        :param h: Height of input image
        """
        super().__init__()
        
        # CPU or GPU
        self.device = device
        
        # Model encoder
        self.encoder = Res18Encoder()
        
        # Weights for the MSE
        # and total variation losses
        self.mse_weight = 0.03
        self.totvar_seg_weight = 0.00003
        self.totvar_det_weight = 0.000002
        
        # Model decoder
        self.decoder = Decoder(int(w/4), int(h/4))
    
    def freeze_encoder(self):
        self.encoder.freeze()
    
    def unfreeze_encoder(self):
        self.encoder.unfreeze()

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
    
    def training_step_detection(self,batch):
        # Unpack batch
        images, targets = batch
        
        # Downsample targets using nearest neighbour method
        downsampled_target =  torch.nn.functional.interpolate(targets, 
                                                              scale_factor=0.25, 
                                                              mode="nearest", 
                                                              recompute_scale_factor=False)
        downsampled_target = to_device(downsampled_target,self.device)
                
        # Run forward pass
        output = self.forward(images, head="detection")
        
        # Compute MSE and total variation losses
        mse_loss = mse_loss_fn(output, downsampled_target)
        ttvar_loss = total_variation(output, [0,1,2])
        
        return (mse_loss + ttvar_loss)
    
    def validation_detection(self,dataloader):
        avg_recall = 0
        avg_precision = 0
        
        for batch_idx,batch in enumerate(dataloader):
            # Unpack batch
            images, targets = batch
            images = to_device(images,self.device)

            # Downsample targets using nearest neighbour method
            downsampled_target = torch.nn.functional.interpolate(targets,
                                                                 scale_factor=0.25,
                                                                 mode="nearest",
                                                                 recompute_scale_factor=False)

            # Run forward pass
            output = self.forward(images,head="detection")
            
            if self.device != 'cpu':
                output = output.cpu()
            
            precision, recall = accuracy(output, downsampled_target)
            
            avg_recall += recall
            avg_precision += precision
                         
        return avg_recall / batch_idx, avg_precision / batch_idx
    
    def training_step_segmentation(self,batch):
        # Unpack batch
        images, targets = batch
        
        # Downsample targets using nearest neighbour method
        downsampled_target = torch.nn.functional.interpolate(targets,
                                                             scale_factor=0.25,
                                                             mode="nearest",
                                                             recompute_scale_factor=False)
        downsampled_target = torch.squeeze(downsampled_target,dim=1) #(batch_size,1,H,W) -> (batch_size,H,W)
        downsampled_target = downsampled_target.type(torch.LongTensor) # convert the target from float to int
        downsampled_target = to_device(downsampled_target,self.device)
        
        # Run forward pass
        output = self.forward(images,head="segmentation")
        
        # Compute output
        softmax = torch.nn.LogSoftmax(dim=1) # wait for hafez reply if it is needed or not
        softmax_output = softmax(output)

        # Compute losse
        nll_loss = torch.nn.NLLLoss()
        #total_loss = nll_loss(softmax_output, downsampled_target) + total_variation_channel(output,0) + total_variation_channel(output,1)
        total_loss = nll_loss(softmax_output, downsampled_target) + total_variation(output, [0,1])

        return total_loss

    def validation_segmentation(self,dataloader):
        correct = 0
        total = 0
        correct_field = 0
        total_field = 0
        correct_lines = 0
        total_lines = 0
        correct_background = 0
        total_background = 0
        tp_field = 0
        pred_field = 0
        fp_field = 0
        fn_field =0
        tp_lines = 0
        pred_lines = 0
        fp_lines = 0
        fn_lines =0
        tp_background = 0
        pred_background = 0
        fp_background = 0
        fn_background =0
        
        for batch_idx,batch in enumerate(dataloader):
            # Unpack batch
            images, targets = batch
            images = to_device(images,self.device)

            # Downsample targets using nearest neighbour method
            downsampled_target = torch.nn.functional.interpolate(targets,
                                                                 scale_factor=0.25,
                                                                 mode="nearest",
                                                                 recompute_scale_factor=False)
            downsampled_target = torch.squeeze(downsampled_target,dim=1) #(batch_size,1,H,W) -> (batch_size,H,W)
            downsampled_target = downsampled_target.type(torch.LongTensor) # convert the target from float to int

            # Run forward pass
            output = self.forward(images,head="segmentation")
            softmax = torch.nn.LogSoftmax(dim=1) # wait for hafez reply if it is needed or not
            softmax_output = softmax(output)
            
            # Get predictions from the maximum value
            _, predicted = torch.max(softmax_output, 1)
            
            if self.device != 'cpu':
                predicted = predicted.cpu()
            
            # Total correct predictions
            correct_predictions = (predicted == downsampled_target)
            correct += correct_predictions.sum().item()
            field_mask = downsampled_target.cpu()==1
            correct_field += (np.logical_and(correct_predictions,field_mask)).sum().item()
            lines_mask = downsampled_target.cpu()==2
            correct_lines +=(np.logical_and(correct_predictions,lines_mask)).sum().item()
            background_mask = downsampled_target.cpu()==0
            correct_background +=(np.logical_and(correct_predictions,background_mask)).sum().item()
                
            # Total number of labels
            total += (downsampled_target.size(0)*downsampled_target.size(1)*downsampled_target.size(2))
            total_field += field_mask.sum().item()
            total_lines += lines_mask.sum().item()
            total_background += background_mask.sum().item()
            
            # Calculate IOU (Double Check)
            # IOU true positive / (true positive + false positive + false negative)
            tp_field += (np.logical_and(correct_predictions,field_mask)).sum().item()
            pred_field = predicted.cpu() == 1
            fp_field += (np.logical_and(pred_field ,np.logical_not(field_mask))).sum().item()
            fn_field += (np.logical_and(np.logical_not(np.logical_and(correct_predictions,field_mask)),field_mask)).sum().item()
            
            
            tp_lines += (np.logical_and(correct_predictions,lines_mask)).sum().item()
            pred_lines = predicted.cpu() == 2
            fp_lines += (np.logical_and(pred_lines ,np.logical_not(lines_mask))).sum().item()
            fn_lines += (np.logical_and(np.logical_not(np.logical_and(correct_predictions,lines_mask)),lines_mask)).sum().item()
            
            tp_background += (np.logical_and(correct_predictions,background_mask)).sum().item()
            pred_background = predicted.cpu() == 0
            fp_background += (np.logical_and(pred_background ,np.logical_not(background_mask))).sum().item()
            fn_background += (np.logical_and(np.logical_not(np.logical_and(correct_predictions,background_mask)),background_mask)).sum().item()
            
            #tn_field = (np.logical_and(np.logical_not(pred_field),np.logical_not(field_mask))).sum().item()# Just as a sanity check
            #print(tn_field+fn_field+tp_field+fp_field)
            #print(iou_field)
    
           
        accuracy = {}    
        accuracy['Total'] = 100 * (correct / total)
        accuracy['Field'] = 100 * (correct_field / total_field)
        accuracy['Lines'] = 100 * (correct_lines / total_lines)
        accuracy['Background'] = 100 * (correct_background / total_background)
        iou = {}
        iou['Field'] = tp_field / (tp_field+fp_field+fn_field)
        iou['Lines'] = tp_lines / (tp_lines+fp_lines+fn_lines)
        iou['Background'] = tp_background / (tp_background+fp_background+fn_background)
        iou['Total'] = (iou['Field'] + iou['Lines'] +iou['Background'])/3
        
             
        return accuracy,iou

