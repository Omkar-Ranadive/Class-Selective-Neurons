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

def forward(model, input_batch, target, class_activations):
    resnet_layers = nn.Sequential(*list(model.children()))

    for index, layer in enumerate(resnet_layers):
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

                        if target[i].item() not in class_activations[index]:
                            class_activations[index].update({target[i].item(): {}})  # ex: {layer_3: {class_0: {} } }
                        
                        if num in class_activations[index][target[i].item()]:
                            class_activations[index][target[i].item()][num] += activation.cpu()
                        else:
                            class_activations[index][target[i].item()].update({num: activation.cpu()})  # ex: {layer_3: {class_0: {bottleneck_0: activation} } }

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

    """
    format --> {
        layer: {
            bottleneck number: [class selectivity per channel]
            }
        }
    """

    class_selectivity = {
        4: {},
        5: {},
        6: {},
        7: {}
    }
    
    # Layer_k = outer layer num, layer_v = dict of the form {class_i: {} ... } 


    for layer_k, layer_v in class_activations.items():
        # for class_k, class_v in class_activations[layer_k].items():
        # For a layer, the number of bottleneck layers will be the same 
        # So, just choose any class (in this case class 0) to get the index of bottleneck layers 
        for bottleneck_k, bottleneck_v in class_activations[layer_k][0].items():
            for class_k in sorted(class_activations[layer_k].keys()):
                if class_k > 0:
                    all_activations_for_this_bottleneck = torch.cat((all_activations_for_this_bottleneck, class_activations[layer_k][class_k][bottleneck_k]), dim=0)
                else:
                    all_activations_for_this_bottleneck = class_activations[layer_k][class_k][bottleneck_k]
            
            all_activations_for_this_bottleneck = all_activations_for_this_bottleneck.t()

            u_max, u_max_indices = torch.max(all_activations_for_this_bottleneck, dim=1)
            u_sum = torch.sum(all_activations_for_this_bottleneck, dim=1)
            u_minus_max = (u_sum - u_max) / (all_activations_for_this_bottleneck.shape[1] - 1)

            selectivity = (u_max - u_minus_max) / (u_max + u_minus_max + epsilon)
            
            class_selectivity[layer_k].update({bottleneck_k: selectivity})
    
    return class_selectivity, class_activations


def calculate_selectivity(data_dir, loader, check_min, check_max): 

    dir = IMGNET_PATH / loader

    # Load pre-trained Resnet 
    if args.arc == 'resnet50':
        model = models.resnet50()
    elif args.arc == 'resnet18': 
        model = models.resnet18()
    elif args.arc == 'resnet34': 
        model = models.resnet34()

    model_dict = model.state_dict() 

    loader_cp = utils.load_imagenet_data(dir=dir, batch_size=256, num_workers=8)

    for cp in range(check_min, check_max+1):
        print(f"Calculating class selctivity for cp {cp}")
        
        cs_dict_path = data_dir / 'cs_dict_{}_cp{}'.format(loader, cp)

        checkpoint = torch.load(data_dir / f'checkpoint_e{cp}.pth.tar')

        # Load checkpoint state dict into the model 
        """
        The key values in checkpoint have different key names, they have an additional "module." in their name 
        Therefore, cleaning the keys before updating state dict of the model 
        """

        for key in checkpoint['state_dict'].keys(): 
            model_key = key.replace("module.", "")
            model_dict[model_key] = checkpoint['state_dict'][key] 

        model.load_state_dict(model_dict)
        model.eval()
        

        if not cs_dict_path.is_file(): 
            class_selectivity, class_activations = get_class_selectivity(model=model, val_loader=loader_cp) 
            utils.save_file(class_selectivity, data_dir / 'cs_dict_{}_cp{}'.format(loader, cp))
            utils.save_file(class_activations, data_dir / 'cs_dict_{}_cp{}_full'.format(loader, cp))


def calculate_selectivity_subcp(data_dir, loader, check_min, check_max): 
    dir = IMGNET_PATH / loader

    # Load pre-trained Resnet 
    if args.arc == 'resnet50':
        model = models.resnet50()
    elif args.arc == 'resnet18': 
        model = models.resnet18()
    elif args.arc == 'resnet34': 
        model = models.resnet34()

    model_dict = model.state_dict() 

    loader_cp = utils.load_imagenet_data(dir=dir, batch_size=256, num_workers=8)

    files = natsort.natsorted(os.listdir(data_dir))
    matches = ['.tar', 'b', 'checkpoint']

    for f in files:
        if all(s in f for s in matches):
            print(f"Calculating class selctivity for file {f}")
            batch_num = re.search('b\d+', f).group()
            cp = re.search('e\d+', f).group()[1:]
            if check_min <= int(cp) <= check_max: 
                cs_dict_path = data_dir / f'cs_dict_{loader}_cp{cp}_{batch_num}'

                checkpoint = torch.load(data_dir / f'checkpoint_e{cp}_{batch_num}.pth.tar')

                # Load checkpoint state dict into the model 
                """
                The key values in checkpoint have different key names, they have an additional "module." in their name 
                Therefore, cleaning the keys before updating state dict of the model 
                """

                for key in checkpoint['state_dict'].keys(): 
                    model_key = key.replace("module.", "")
                    model_dict[model_key] = checkpoint['state_dict'][key] 

                model.load_state_dict(model_dict)
                model.eval()
                

                if not cs_dict_path.is_file(): 
                    class_selectivity, class_activations = get_class_selectivity(model=model, val_loader=loader_cp) 
                    utils.save_file(class_selectivity, data_dir / f'cs_dict_{loader}_cp{cp}_{batch_num}')
                    utils.save_file(class_activations, data_dir / f'cs_dict_{loader}_cp{cp}_{batch_num}_full')


if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--img_dir", type=str, default=IMGNET_PATH)
    parser.add_argument("--loader", default='val', type=str)
    parser.add_argument("--arc", default='resnet50', type=str)
    parser.add_argument("--check_min", type=int, default=0)
    parser.add_argument("--check_max", type=int, default=90)
    parser.add_argument("--sub", action='store_true', help="Use this for sub-checkpointing scenario")
    args = parser.parse_args()

    data_dir = DATA_PATH / args.data_dir

    if not args.sub: 
        calculate_selectivity(data_dir, args.loader, args.check_min, args.check_max)
    else: 
        calculate_selectivity_subcp(data_dir, args.loader, args.check_min, args.check_max)

  

