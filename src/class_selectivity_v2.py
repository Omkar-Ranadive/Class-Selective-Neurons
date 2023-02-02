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
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor


def forward(input_batch, target, class_activations, return_nodes, feature_extractor): 
    out = feature_extractor(input_batch)
    vals = list(return_nodes.values()) 
    prev = vals[0] 
    counter = 0 
    nums = []
    for val in vals: 
        if val != prev: 
            counter = 0 
        
        nums.append(counter)
        counter += 1 
        prev = val 
    
    nums = dict(zip(return_nodes.keys(), nums))
    targets = target.cpu().numpy().tolist()

    for k, v in out.items(): 
        # Key is the node name, value is the actual feature, i.e, activation output tensor  
        index = return_nodes[k] 
        activations = torch.mean(v.view(v.size(0), v.size(1), -1), dim=2).cpu().numpy()

        for i, activation in enumerate(activations): 
            activation = np.expand_dims(activation, axis=0)
            num = nums[k]

            if targets[i] not in class_activations[index]:
                class_activations[index].update({targets[i]: {}})  # ex: {layer_3: {class_0: {} } }
            
            if num in class_activations[index][targets[i]]:
                class_activations[index][targets[i]][num] += activation
            else:
                class_activations[index][targets[i]].update({num: activation})  # ex: {layer_3: {class_0: {bottleneck_0: activation} } }
   
    return input_batch, class_activations 


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

    if args.arc == 'resnet50': 
        return_nodes = {
                        'layer1.0.relu_2': 4, 'layer1.1.relu_2': 4,'layer1.2.relu_2': 4,
                        'layer2.0.relu_2': 5,'layer2.1.relu_2': 5, 'layer2.2.relu_2': 5, 'layer2.3.relu_2': 5,
                        'layer3.0.relu_2': 6, 'layer3.1.relu_2':6, 'layer3.2.relu_2': 6, 'layer3.3.relu_2': 6, 'layer3.4.relu_2': 6, 'layer3.5.relu_2': 6,
                        'layer4.0.relu_2': 7, 'layer4.1.relu_2': 7, 'layer4.2.relu_2': 7
                    }
    elif args.arc == 'resnet18': 
        return_nodes = {
                        'layer1.0.relu_1': 4, 'layer1.1.relu_1': 4,
                        'layer2.0.relu_1': 5, 'layer2.1.relu_1': 5,
                        'layer3.0.relu_1': 6, 'layer3.1.relu_1': 6,
                        'layer4.0.relu_1': 7, 'layer4.1.relu_1': 7
                    }
    elif args.arc == 'resnet34': 
        return_nodes = {
                        'layer1.0.relu_1': 4, 'layer1.1.relu_1': 4, 'layer1.2.relu_1': 4,
                        'layer2.0.relu_1': 5, 'layer2.1.relu_1': 5, 'layer2.2.relu_1': 5, 'layer2.3.relu_1': 5,
                        'layer3.0.relu_1': 6, 'layer3.1.relu_1': 6, 'layer3.2.relu_1': 6, 'layer3.3.relu_1': 6, 'layer3.4.relu_1': 6, 'layer3.5.relu_1': 6,
                        'layer4.0.relu_1': 7, 'layer4.1.relu_1': 7, 'layer4.2.relu_1': 7
                    }               

    feature_extractor = create_feature_extractor(model, return_nodes=list(return_nodes.keys()))


    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm(enumerate(val_loader)):
        
            if torch.cuda.is_available():
                target = target.to('cuda')
                images = images.to('cuda')

            input_batch, class_activations = forward(images, target, class_activations, return_nodes, feature_extractor)
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
                    all_activations_for_this_bottleneck = np.concatenate((all_activations_for_this_bottleneck, class_activations[layer_k][class_k][bottleneck_k]), axis=0)
                else:
                    all_activations_for_this_bottleneck = class_activations[layer_k][class_k][bottleneck_k]
            
            all_activations_for_this_bottleneck = all_activations_for_this_bottleneck.T

            u_max = np.max(all_activations_for_this_bottleneck, axis=1)
            u_sum = np.sum(all_activations_for_this_bottleneck, axis=1)
            u_minus_max = (u_sum - u_max) / (all_activations_for_this_bottleneck.shape[1] - 1)

            selectivity = (u_max - u_minus_max) / (u_max + u_minus_max + epsilon)
            
            class_selectivity[layer_k].update({bottleneck_k: selectivity})
    
    return class_selectivity, class_activations


def forward_grad(features, targets, class_activations, return_nodes): 
    """
    Same as forward function but this one will keep track of gradients (slower and consumes more memory) - necessary to tune models w.r.t selectivity 
    """
    vals = list(return_nodes.values()) 
    prev = vals[0] 
    counter = 0 
    nums = []
    for val in vals: 
        if val != prev: 
            counter = 0 
        
        nums.append(counter)
        counter += 1 
        prev = val 
    
    nums = dict(zip(return_nodes.keys(), nums))

    for k, v in features.items(): 
        # Key is the node name, value is the actual feature, i.e, activation output tensor  
        index = return_nodes[k] 
        activations = torch.mean(v.view(v.size(0), v.size(1), -1), dim=2)

        for i, activation in enumerate(activations): 
            activation = torch.unsqueeze(activation, dim=0)
            num = nums[k]

            if targets[i] not in class_activations[index]:
                class_activations[index].update({targets[i]: {}})  # ex: {layer_3: {class_0: {} } }
            
            if num in class_activations[index][targets[i]]:
                class_activations[index][targets[i]][num] += activation
            else:
                class_activations[index][targets[i]].update({num: activation})  # ex: {layer_3: {class_0: {bottleneck_0: activation} } }
   
    return class_activations 

    
    
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
            model_key = model_key.replace("model.", "")
            model_dict[model_key] = checkpoint['state_dict'][key] 

        model.load_state_dict(model_dict)
        model.eval()
        


        if not cs_dict_path.is_file(): 
            class_selectivity, class_activations = get_class_selectivity(model=model, val_loader=loader_cp) 
            utils.save_file(class_selectivity, data_dir / 'cs_dict_{}_cp{}'.format(loader, cp))
            utils.save_file(class_activations, data_dir / 'cs_dict_{}_cp{}_full'.format(loader, cp))


if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--img_dir", type=str, default=IMGNET_PATH)
    parser.add_argument("--loader", default='val', type=str)
    parser.add_argument("--arc", default='resnet50', type=str)
    parser.add_argument("--check_min", type=int, default=0)
    parser.add_argument("--check_max", type=int, default=90)
    args = parser.parse_args()

    data_dir = DATA_PATH / args.exp_name

    calculate_selectivity(data_dir, args.loader, args.check_min, args.check_max)
    
    # model = models.resnet50(pretrained=True)
    # loader = 'val'
    # loader_cp = utils.load_imagenet_data(dir=IMGNET_PATH / loader, batch_size=128, num_workers=8)
    # class_selectivity, class_activations = get_class_selectivity(model=model, val_loader=loader_cp) 
    # utils.save_file(class_selectivity, data_dir / 'cs_dict_r50_new')
    # utils.save_file(class_activations, data_dir / 'cs_dict_r50_new_full')
