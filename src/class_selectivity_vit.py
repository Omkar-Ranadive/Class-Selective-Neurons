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
import copy 
import argparse
import natsort 
import re 
import os 
import sys 
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from  timm.models.vision_transformer import Block 
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor


def forward(model, feature_extractor, input_batch, target, class_activations):    
    node_to_mod = {
    "blocks.0.mlp.act": 4,
    "blocks.1.mlp.act": 4,
    "blocks.2.mlp.act": 4,
    "blocks.3.mlp.act": 5, 
    "blocks.4.mlp.act": 5, 
    "blocks.5.mlp.act": 5, 
    "blocks.6.mlp.act": 6, 
    "blocks.7.mlp.act": 6, 
    "blocks.8.mlp.act": 6, 
    "blocks.9.mlp.act": 7, 
    "blocks.10.mlp.act": 7, 
    "blocks.11.mlp.act": 9
    }

    out = feature_extractor(input_batch)
    for k, v in out.items():
        # Key is the node name, value is the actual feature, i.e, activation output tensor  
        index = node_to_mod[k] 
        activations = torch.mean(v.view(v.size(0), v.size(1), -1), dim=2)
     
        for i, activation in enumerate(activations): 
            activation = torch.abs(activation)   # As it's GeLu, it may not be positive, so limit value between 0, x 
            activation = torch.unsqueeze(activation, dim=0)

            if target[i].item() not in class_activations[index]:
                class_activations[index].update({target[i].item(): {}})  # ex: {layer_3: {class_0: {} } }
            
            if k in class_activations[index][target[i].item()]:
                class_activations[index][target[i].item()][k] += activation.cpu()
            else:
                class_activations[index][target[i].item()].update({k: activation.cpu()})  # ex: {layer_3: {class_0: {bottleneck_0: activation} } }
        

def get_class_activations(model, val_loader):
    # switch to evaluate mode
    return_nodes = ["blocks.0.mlp.act","blocks.1.mlp.act","blocks.2.mlp.act","blocks.3.mlp.act","blocks.4.mlp.act","blocks.5.mlp.act",
    "blocks.6.mlp.act","blocks.7.mlp.act","blocks.8.mlp.act","blocks.9.mlp.act","blocks.10.mlp.act","blocks.11.mlp.act"]
   
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

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
        7: {}, 
        9: {}
    }
    counter = 0
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm(enumerate(val_loader)):
        
            if torch.cuda.is_available():
                target = target.to('cuda')
                images = images.to('cuda')

            output = forward(model, feature_extractor, images, target, class_activations)
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
        7: {}, 
        9: {}
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
    # model = models.resnet50()
    # model_dict = model.state_dict() 
    model = timm.create_model('vit_small_patch16_224', pretrained=True)
    cp = 90

    loader_cp = utils.load_imagenet_data(dir=dir, batch_size=256, num_workers=4)
    cs_dict_path = data_dir / 'cs_dict_{}_cp{}'.format(loader, cp)

    model.eval()
    if not cs_dict_path.is_file(): 
        class_selectivity, class_activations = get_class_selectivity(model=model, val_loader=loader_cp) 
        utils.save_file(class_selectivity, data_dir / 'cs_dict_{}_cp{}'.format(loader, cp))
        utils.save_file(class_activations, data_dir / 'cs_dict_{}_cp{}_full'.format(loader, cp))



    # for cp in range(check_min, check_max+1):
    #     print(f"Calculating class selctivity for cp {cp}")
        
    #     cs_dict_path = data_dir / 'cs_dict_{}_cp{}'.format(loader, cp)

    #     checkpoint = torch.load(data_dir / f'checkpoint_e{cp}.pth.tar')

    #     # Load checkpoint state dict into the model 
    #     """
    #     The key values in checkpoint have different key names, they have an additional "module." in their name 
    #     Therefore, cleaning the keys before updating state dict of the model 
    #     """

    #     for key in checkpoint['state_dict'].keys(): 
    #         model_key = key.replace("module.", "")
    #         model_dict[model_key] = checkpoint['state_dict'][key] 

    #     model.load_state_dict(model_dict)
    #     model.eval()
        

    #     if not cs_dict_path.is_file(): 
    #         class_selectivity, class_activations = get_class_selectivity(model=model, val_loader=loader_cp) 
    #         utils.save_file(class_selectivity, data_dir / 'cs_dict_{}_cp{}'.format(loader, cp))
    #         utils.save_file(class_activations, data_dir / 'cs_dict_{}_cp{}_full'.format(loader, cp))


if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--img_dir", type=str, default=IMGNET_PATH)
    parser.add_argument("--loader", default='val', type=str)
    parser.add_argument("--check_min", type=int, default=0)
    parser.add_argument("--check_max", type=int, default=90)
    args = parser.parse_args()

    data_dir = DATA_PATH / args.data_dir

    calculate_selectivity(data_dir, args.loader, args.check_min, args.check_max)

    # cs_dict = utils.load_file(data_dir / 'cs_dict_val_cp90') 
    # class_activations = utils.load_file(data_dir / 'cs_dict_val_cp90_full')

    # layers = [4, 5, 6, 7] 

    # for l in layers: 
    #     for block, v in cs_dict[l].items(): 
    #         print(f"Layer: {l} Block {block}", v.shape)


    # print("*"*20)
    # epsilon = 1e-6

    # for layer_k, layer_v in class_activations.items():
    # # for class_k, class_v in class_activations[layer_k].items():
    # # For a layer, the number of bottleneck layers will be the same 
    # # So, just choose any class (in this case class 0) to get the index of bottleneck layers 
    #     for bottleneck_k, bottleneck_v in class_activations[layer_k][0].items():
    #         for class_k in sorted(class_activations[layer_k].keys()):
    #             if class_k > 0:
    #                 all_activations_for_this_bottleneck = torch.cat((all_activations_for_this_bottleneck, class_activations[layer_k][class_k][bottleneck_k]), dim=0)
    #             else:
    #                 all_activations_for_this_bottleneck = class_activations[layer_k][class_k][bottleneck_k]
            
    #         all_activations_for_this_bottleneck = all_activations_for_this_bottleneck.t()
    #         print("Bottleneck: ", bottleneck_k)
    #         print(all_activations_for_this_bottleneck.shape)

    #         u_max, u_max_indices = torch.max(all_activations_for_this_bottleneck, dim=1)
    #         print("umax:", u_max.shape)
    #         u_sum = torch.sum(all_activations_for_this_bottleneck, dim=1)
    #         u_minus_max = (u_sum - u_max) / (all_activations_for_this_bottleneck.shape[1] - 1)

    #         selectivity = (u_max - u_minus_max) / (u_max + u_minus_max + epsilon)

    #         print("*"*20)

            


    # print(timm.list_models('*vit*'))

#     model = timm.create_model('vit_small_patch16_224', pretrained=True)
#     # model = models.resnet101()
#     model.eval()

#     # train_nodes, eval_nodes = get_graph_node_names(model)

#     return_nodes = ["blocks.0.mlp.act","blocks.1.mlp.act","blocks.2.mlp.act","blocks.3.mlp.act","blocks.4.mlp.act","blocks.5.mlp.act",
#     "blocks.6.mlp.act","blocks.7.mlp.act","blocks.8.mlp.act","blocks.9.mlp.act","blocks.10.mlp.act","blocks.11.mlp.act"]

#     node_to_mod = {
# "blocks.0.mlp.act": 4,
# "blocks.1.mlp.act": 4,
# "blocks.2.mlp.act": 4,
# "blocks.3.mlp.act": 5, 
# "blocks.4.mlp.act": 5, 
# "blocks.5.mlp.act": 5, 
# "blocks.6.mlp.act": 6, 
# "blocks.7.mlp.act": 6, 
# "blocks.8.mlp.act": 6, 
# "blocks.9.mlp.act": 7, 
# "blocks.10.mlp.act": 7, 
# "blocks.11.mlp.act": 9
#     }

#     feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)


#     with torch.no_grad():
#         out = feature_extractor(torch.zeros(50, 3, 224, 224))
    
#     print(out.keys())

    # vit_layers = nn.Sequential(*list(model.children()))
    
    # for index, layer in enumerate(vit_layers): 
    #     if isinstance(layer, torch.nn.modules.container.Sequential):
    #         for num, child in enumerate(layer.children()): 
    #             if isinstance(child, Block): 
    #                 if hasattr(child, 'act'):
    #                     print(child.act)




                    
  