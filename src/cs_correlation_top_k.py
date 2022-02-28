from constants import DATA_PATH, EXP_PATH, IMGNET_PATH
from class_selectivity import bottleneck_layer
import utils

import torch
from torch import nn
from torch.nn.modules.activation import ReLU
import torchvision.models as models
from torchvision.models.resnet import Bottleneck as Bottleneck
import argparse 
import os 
import logging 
from collections import defaultdict 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from itertools import combinations
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--check_num", required=True, type=int)
# parser.add_argument("--check_min", required=True, type=int)
# parser.add_argument("--check_max", required=True, type=int)
parser.add_argument("--k", default=3, type=int)
parser.add_argument("--num_workers", default=8, type=int)
args = parser.parse_args()


EXP_DIR = EXP_PATH / args.exp_name
os.makedirs(EXP_DIR, exist_ok=True)

# Setup logger 
logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info.log'), format='%(message)s', filemode='w')
logger=logging.getLogger() 


# Load imagenet categories 
with open(DATA_PATH / "imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

num_classes = 1000 
 # Assign each class a different color (1000 color palette)
colors = sns.color_palette('viridis', num_classes) 
color_dict = dict(zip(range(0, num_classes), colors))


selectivities = {}
for layer in [4, 5, 6, 7]: 
    selectivities[layer] = defaultdict(list)

# Load pre-trained Resnet 
model = models.resnet50()
model_dict = model.state_dict() 


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
                    activations = torch.mean(input_batch.view(input_batch.size(0), input_batch.size(1), -1), dim=2)

                    # print(activations.shape)
                    # exit()

                    for i, activation in enumerate(activations): 
                        activation = torch.unsqueeze(activation, dim=0)

                        if target[i].item() not in class_activations[index]:
                            class_activations[index].update({target[i].item(): {}})  # ex: {layer_3: {class_0: {} } }
                        
                        if num in class_activations[index][target[i].item()]:
                            # class_activations[index][target[i].item()][num] += activation.cpu()
                            class_activations[index][target[i].item()][num] = torch.cat((class_activations[index][target[i].item()][num], activation.cpu()), dim=0)
                        else:
                            class_activations[index][target[i].item()].update({num: activation.cpu()})  # ex: {layer_3: {class_0: {bottleneck_0: activation} } }

                else: 
                    input_batch = child(input_batch)

        else:
            input_batch = layer(input_batch)

    return input_batch


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
    class_counter = defaultdict(int)
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm(enumerate(val_loader)):
            # for t in target:
            #     print(t.item())
            #     class_counter[t.item()] += 1
            # print("--------------------")
            # for k, v in class_counter.items():
            #         print(k, ":", v)
            # class_counter[target.item()] += 1
            # if class_counter[target.item()] > 100:
            #     continue
            if torch.cuda.is_available():
                target = target.to('cuda')
                images = images.to('cuda')

            output = forward(model, images, target, class_activations)
            # if i>2:
            #     print("-----------------------------------------------------------------------")
            #     for k, v in class_counter.items():
            #         print(k, ":", v)
            #     exit()
            # if counter > 110:
                # break
            counter += 1
    
    return class_activations

# Nikhil (02/26/2022): Hardcoding checkpoint numbers for now.
# I want plots for start, middle, and end of training.
# for cp in range(args.check_num, args.check_num+1):
for cp in [1, 45, 89]:
    logger.info("-"*20)
    logger.info("Checkpoint {}".format(cp))
    # class_activations = utils.load_file(DATA_PATH / 'cs_dict_val_cp{}_full'.format(cp))
    checkpoint = torch.load(DATA_PATH / 'model_checkpoints' / 'CHECKPOINTS' / 'EXPE1' / 'checkpoint_epoch{}.pth.tar'.format(cp))
    # Layer -> Class -> Bottleneck

    val_dir = IMGNET_PATH / 'val'
    val_loader = utils.load_imagenet_data(dir=val_dir, batch_size=500, num_workers=args.num_workers)

    for key in checkpoint['state_dict'].keys(): 
        model_key = key.replace("module.", "")
        model_dict[model_key] = checkpoint['state_dict'][key] 

    model.load_state_dict(model_dict)
    start = time.time()
    class_activations = get_class_activations(model, val_loader)
    print("Time taken:", time.time() - start)
    # print(class_activations.keys())
    # print(class_activations[4].keys())
    # print(class_activations[4][0].keys())
    # print(class_activations[4][0][0].shape)
    # exit()

    epsilon = 1e-6

    act_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # k -> class -> Layer_k -> [[bn1], [bn2]...] where bni includes bottleneck values for each class [1000d mat]
    class_counter = {}
    # Layer_k = outer layer num, layer_v = dict of the form {class_i: {} ... }
    print("Calculating Activation values...")
    start = time.time()
    for layer_k, layer_v in class_activations.items():
        # class_counter[layer_k] = defaultdict(int)
        # for class_k, class_v in class_activations[layer_k].items():
        # For a layer, the number of bottleneck layers will be the same 
        # So, just choose any class (in this case class 0) to get the index of bottleneck layers
        for class_k, class_v in class_activations[layer_k].items():
            for bottleneck_k, bottleneck_v in class_activations[layer_k][class_k].items():
                # for class_k in sorted(class_activations[layer_k].keys()):
                #     if class_k > 0:
                #         all_activations_for_this_bottleneck = torch.cat((all_activations_for_this_bottleneck, class_activations[layer_k][class_k][bottleneck_k]), dim=0)
                #     else:
                #         all_activations_for_this_bottleneck = class_activations[layer_k][class_k][bottleneck_k]
                # print(class_activations[layer_k][class_k][bottleneck_k].shape)
                # exit()
                for k in range(1, args.k+1):
                    top_k_activations_for_this_bottleneck, _ = torch.topk(class_activations[layer_k][class_k][bottleneck_k], k=k)
                    act_vals[k][class_k][layer_k].append(torch.mean(top_k_activations_for_this_bottleneck, dim=1).numpy())
                # all_activations_for_this_bottleneck = all_activations_for_this_bottleneck.t()
                # print(all_activations_for_this_bottleneck.shape)
                # avg_activations_for_bn = torch.mean(all_activations_for_this_bottleneck, dim=0) 
                
                # act_vals[layer_k].append(avg_activations_for_bn.numpy())

                # u_max_k, u_max_k_indices = torch.topk(avg_activations_for_bn, k=args.k)

                # for i, ci in enumerate(u_max_k_indices.numpy()): 
                #     class_counter[layer_k][ci] += 1 


                # # Keep track of class selective index for bottleneck layers 
                # u_max, u_max_indices = torch.max(all_activations_for_this_bottleneck, dim=1)
                # u_sum = torch.sum(all_activations_for_this_bottleneck, dim=1)
                # u_minus_max = (u_sum - u_max) / (all_activations_for_this_bottleneck.shape[1] - 1)

                # selectivity = (u_max - u_minus_max) / (u_max + u_minus_max + epsilon)

                # avg_selectivity = torch.mean(selectivity)

                # # Selectivity at cp 0 will skew the result 
                # if cp > 0: 
                #     selectivities[layer_k][bottleneck_k].append(avg_selectivity.item())
    
    print("Activation values calculated...")
    print("Time taken: {}".format(time.time() - start))

    # Save layer wise info to log 
    # for layer, counts in class_counter.items(): 
    #     logger.info("*"*20)
    #     logger.info("Layer {}".format(layer))  

    #     for k, v in counts.items(): 
    #         logger.info("ID {} Name {} Count: {}".format(k, categories[k], v))



    # df_counts = pd.DataFrame(class_counter).T
    # # bp = sns.barplot(data=df_counts) 
   
    # ax = df_counts.plot.bar(rot=0, stacked=True, legend=False, color=color_dict)
    # ax.set_title("CP {}: Top categories by layer".format(cp))
    # # # for c in ax.containers:
    # # #     # Optional: if the segment is small or 0, customize the labels
    # # #     labels = df_counts[l]
    # # #     # labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
    # # #     # remove the labels parameter if it's not needed for customized labels
    # # #     ax.bar_label(c, labels=labels, label_type='center')

    # plt.savefig(EXP_DIR / 'cp{}_categories_bar.jpg'.format(cp))
    # plt.clf()   
    # plt.close()


    """
    Form a dataframe from the dict 
    Each column is length 1000 of the form lk_bi for k = layer num and i = bottleneck num 
    """

    # df = pd.DataFrame()
    # df_dict = defaultdict(lambda: defaultdict(dict))
    # df_sv = pd.DataFrame() # For selectivity
    print("Plotting...")

    for k, v in sorted(act_vals.items()):
        for class_k, class_v in sorted(act_vals[k].items()):
            df = pd.DataFrame()
            for layer, bns in sorted(act_vals[k][class_k].items()):
                for i, bn_val in enumerate(bns): 
                    df["l{}_b{}".format(layer, i)] = bn_val 
            
            # df_dict[k][class_k] = df
            SAVE_DIR = EXP_DIR / 'k.{}'.format(k) / '{}.{}'.format(class_k, categories[class_k])
            SAVE_DIR.mkdir(parents=True, exist_ok=True)

            # Plot for all layers
            # hm = sns.heatmap(df_dict[k][class_k].corr())
            hm = sns.heatmap(df.corr())
            hm.set(title="CP {} all layers for {}".format(cp, categories[class_k]))
            
            SAVE_PATH = SAVE_DIR / 'cp{}_all_layers.jpg'.format(cp)
            
            plt.savefig(SAVE_PATH)
            plt.clf()
            plt.close()

            # Plot two layers at a time. Layers are always the same. So choosing k=1 and class = 0.
            layers = combinations(act_vals[1][0].keys(), 2)
            
            for li, lj in layers: 
                cols_i = ["l{}_b{}".format(li, i) for i in range(len(act_vals[1][0][li]))] 
                cols_j = ["l{}_b{}".format(lj, j) for j in range(len(act_vals[1][0][lj]))] 

                # df_sliced = df_dict[k][class_k][cols_i + cols_j]
                df_sliced = df[cols_i + cols_j]

                # Plot for the sliced frame 
                hm_sliced = sns.heatmap(df_sliced.corr(), annot=True)
                hm_sliced.set(title="CP {} Layers {} - {} for {}".format(cp, li, lj, categories[class_k]))
                
                SAVE_PATH = SAVE_DIR / 'cp{}_layers_{}_{}.jpg'.format(cp, li, lj)
                
                plt.savefig(SAVE_PATH)
                plt.clf()
            
            plt.close()
        

    # Plot for all layers
    # hm = sns.heatmap(df.corr())
    # hm.set(title="CP {} all layers".format(cp))
    # plt.savefig(EXP_DIR / 'cp{}_all_layers.jpg'.format(cp))
    # plt.clf()
    # plt.close()

    # # Plot two layers at a time. Layers are always the same. So choosing k=1 and class = 0.
    # layers = combinations(act_vals[1][0].keys(), 2)
    
    # for li, lj in layers: 
    #     cols_i = ["l{}_b{}".format(li, i) for i in range(len(act_vals[li]))] 
    #     cols_j = ["l{}_b{}".format(lj, j) for j in range(len(act_vals[lj]))] 

    #     df_sliced = df[cols_i + cols_j]

    #     # Plot for the sliced frame 
    #     hm_sliced = sns.heatmap(df_sliced.corr(), annot=True)
    #     hm_sliced.set(title="CP {} Layers {} - {} for {}".format(cp, li, lj, categories[class_k]))
        
    #     SAVE_DIR = EXP_DIR / 'k.{}'.format(k) / '{}.{}'.format(class_k, categories[class_k])
    #     SAVE_DIR.mkdir(parents=True, exist_ok=True)
    #     SAVE_PATH = SAVE_DIR / 'cp{}_layers_{}_{}.jpg'.format(cp, li, lj)
        
    #     plt.savefig(SAVE_PATH)
    #     plt.clf()
    
    # plt.close()

# df_sv = pd.DataFrame()
# for layer, bns in sorted(selectivities.items()): 
#     for k, bn_vals in sorted(bns.items()): 
#         df_sv["l{}_b{}".format(layer, k)] = bn_vals 


# hm = sns.heatmap(df_sv.corr())
# hm.set(title="All checkpoints selectivity")
# plt.savefig(EXP_DIR / 'All_checkpoints_selectivity_corr.jpg')
# plt.clf()
# plt.close()


# sns.lineplot(data=df_sv)
# plt.savefig(EXP_DIR / 'All_checkpoints_selectivity_plot.jpg')
# plt.clf()
# plt.close()