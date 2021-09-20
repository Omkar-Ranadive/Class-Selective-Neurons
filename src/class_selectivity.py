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
import utils 
import time
from tqdm import tqdm

def forward(model, input_batch, target, class_activations):
    resnet_layers = nn.Sequential(*list(model.children()))

    for index, layer in enumerate(resnet_layers):

        if isinstance(layer, torch.nn.modules.linear.Linear):
            # Flatten input batch once linear layer is reached 
            input_batch = torch.flatten(input_batch, start_dim=1)
        
        # If the present layer is sequential, go deeper 
        if isinstance(layer, torch.nn.modules.container.Sequential):
            
            for num, child in enumerate(layer.children()): 
                if isinstance(child, Bottleneck): 
                    input_batch = bottleneck_layer(input_batch, child)
                    activation = torch.mean(input_batch.view(input_batch.size(0), input_batch.size(1), -1), dim=2)

                    if target[0].item() not in class_activations[index]:
                        class_activations[index].update({target[0].item(): {}})  # ex: {layer_3: {class_0: {} } }
                    
                    if num in class_activations[index][target[0].item()]:
                        class_activations[index][target[0].item()][num] += activation
                    else:
                        class_activations[index][target[0].item()].update({num: activation})  # ex: {layer_3: {class_0: {bottleneck_0: activation} } }

                    # for k, v in class_activations.items():
                    #     print(k, ":", v)
                    #     for k2, v2 in class_activations[k].items():
                    #         print(k2, ":", v2)
                    #         for k3, v3 in class_activations[k][k2].items():
                    #             print(k3, ":", v3)
                    #             print("Length:", v3.shape)
                    # exit()
                else: 
                    input_batch = child(input_batch)

        else:
            input_batch = layer(input_batch)

    return input_batch


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


def get_class_activations(model, val_loader):
    # switch to evaluate mode
    model.eval()
    model.to('cuda')
    """
    format --> {
        layer: {
            class: {
                bottleneck number: [activations]
                }
            }
        }
    """
    class_activations = {
        4: {},
        5: {},
        6: {},
        7: {}
    }
    counter = 0
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm(enumerate(val_loader)):
        
            if torch.cuda.is_available():
                target = target.to('cuda')
                images = images.to('cuda')

            output = forward(model, images, target, class_activations)
            # if counter > 110:
            #     break
            counter += 1
    
    return class_activations


def get_class_selectivity(model, val_loader):
    epsilon = 1e-6
    class_activations = get_class_activations(model, val_loader)

    class_selectivity = {
        4: {},
        5: {},
        6: {},
        7: {}
    }

    for layer_k, layer_v in class_activations.items():
        # for class_k, class_v in class_activations[layer_k].items():
        for bottleneck_k, bottleneck_v in class_activations[layer_k][0].items():
            for class_k in class_activations[layer_k].keys():
                if class_k > 0:
                    all_activations_for_this_bottleneck = torch.cat((all_activations_for_this_bottleneck, class_activations[layer_k][class_k][bottleneck_k]), dim=0)
                else:
                    all_activations_for_this_bottleneck = class_activations[layer_k][class_k][bottleneck_k]
            
            all_activations_for_this_bottleneck = all_activations_for_this_bottleneck.t()

            u_max, u_max_indices = torch.max(all_activations_for_this_bottleneck, dim=1)
            u_sum = torch.sum(all_activations_for_this_bottleneck, dim=1)
            u_minus_max = (u_sum - u_max) / all_activations_for_this_bottleneck.shape[1]

            selectivity = (u_max - u_minus_max) / (u_max + u_minus_max + epsilon)
            
            class_selectivity[layer_k].update({bottleneck_k: selectivity})
    
    return class_selectivity


if __name__ == "__main__":
    model = models.resnet50(pretrained=True)
    
    val_dir = IMGNET_PATH / 'val'

    # Prepare validation loader
    val_loader = utils.load_imagenet_data(dir=val_dir, batch_size=1, num_workers=8)

    get_class_selectivity(model=model, val_loader=val_loader) 
    
    