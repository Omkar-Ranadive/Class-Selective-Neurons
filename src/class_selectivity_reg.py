import torch
from torch import nn
from torch.nn.modules.activation import ReLU
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import torchvision.datasets as datasets
from torchvision.models.resnet import resnet50
from constants import DATA_PATH, IMGNET_PATH
import numpy as np 
from torchvision.models.resnet import Bottleneck as Bottleneck
from torchvision.models.resnet import BasicBlock as BasicBlock
import utils 
import time
from tqdm import tqdm
import copy 
import argparse
import natsort 
import re 
import os 
import sys 


def bottleneck_layer(input_batch, child):
    # The bottle neck layers also follow this structure 
    # Reference:  https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html 
    identity = input_batch

    out = child.conv1(input_batch)
    out = child.bn1(out)
    out = child.relu(out)
    
    # if ablate: 
    #     out = zero_out_activation(out, num_channels)

    out = child.conv2(out)
    out = child.bn2(out)
    out = child.relu(out)

    out = child.conv3(out)
    out = child.bn3(out)

    if child.downsample is not None:
        identity = child.downsample(input_batch)

    out += identity
    out = child.relu(out) 

    # if ablate: 
    #     out = zero_out_activation(out, num_channels)

    return out


def basic_layer(input_batch, child):
    identity = input_batch

    out = child.conv1(input_batch)
    out = child.bn1(out)
    out = child.relu(out)

    out = child.conv2(out)
    out = child.bn2(out)

    if child.downsample is not None:
        identity = child.downsample(input_batch)

    out += identity
    out = child.relu(out)

    return out


def get_selectivity_grad(index, layer, class_activations, input_batch, target): 
    if isinstance(layer, torch.nn.modules.linear.Linear):
        # Flatten input batch once linear layer is reached 
        input_batch = torch.flatten(input_batch, start_dim=1)
    
    # If the present layer is sequential, go deeper 
    if isinstance(layer, torch.nn.modules.container.Sequential):
        for num, child in enumerate(layer.children()): 
            if isinstance(child, Bottleneck) or isinstance(child, BasicBlock):
                input_batch = bottleneck_layer(input_batch, child) if isinstance(child, Bottleneck) else basic_layer(input_batch, child)
                activations = torch.mean(input_batch.view(input_batch.size(0), input_batch.size(1), -1), dim=2)

                for i, activation in enumerate(activations): 
                    activation = torch.unsqueeze(activation, dim=0)
                    class_index = target[i].item() 

                    if class_index not in class_activations[index]:
                        class_activations[index].update({class_index: {}})  # ex: {layer_3: {class_0: {} } }
                    
                    if num in class_activations[index][class_index]:
                        class_activations[index][class_index][num] += activation.cpu()
                    else:
                        class_activations[index][class_index].update({num: activation.cpu()})  # ex: {layer_3: {class_0: {bottleneck_0: activation} } }

            else: 
                input_batch = child(input_batch)

    else:
        input_batch = layer(input_batch)

    return input_batch, class_activations
        