import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid, save_image
import torch.nn as nn
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
from torchvision import datasets, transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_result(model, target_layer = None, image = None):
    '''
        Needs the model, the layer over which it will happen.
        It also needs an torch image of correct dimension.
    '''
    if target_layer == None:
        target_layer = model.conv2
    gradcam = GradCAM(model, target_layer)
    print(image.shape)
    mask, _ = gradcam(image)
    heatmap, result = visualize_cam(mask, image)
    
    return heatmap, result