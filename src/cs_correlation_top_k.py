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
import sys 
import dill 

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
# parser.add_argument("--check_num", required=True, type=int)
# parser.add_argument("--check_min", required=True, type=int)
# parser.add_argument("--check_max", required=True, type=int)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--k", default=3, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--plot_mat", action='store_true')
parser.add_argument("--cd", default=None, type=str, help="Can provide the correlation dictionary directly if available")
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
colors = sns.color_palette("flare", num_classes)
# Add alpha to the palette 
alpha = 0.2
colors = [(*color, alpha) for color in colors]



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
            if torch.cuda.is_available():
                target = target.to('cuda')
                images = images.to('cuda')

            output = forward(model, images, target, class_activations)
            counter += 1
    
    return class_activations

# Nikhil (02/26/2022): Hardcoding checkpoint numbers for now.
# I want plots for start, middle, and end of training.
# for cp in range(args.check_num, args.check_num+1):

checkpoints = range(1, 90, 4)  # Step by 4 

if not args.cd: 
    corr_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for cp in checkpoints:
        logger.info("-"*20)
        logger.info("Checkpoint {}".format(cp))
        # class_activations = utils.load_file(DATA_PATH / 'cs_dict_val_cp{}_full'.format(cp))
        checkpoint = torch.load(DATA_PATH / 'model_checkpoints' / 'CHECKPOINTS' / 'EXPE1' / 'checkpoint_epoch{}.pth.tar'.format(cp))
        # Layer -> Class -> Bottleneck

        val_dir = IMGNET_PATH / 'val'
        val_loader = utils.load_imagenet_data(dir=val_dir, batch_size=args.batch_size, num_workers=args.num_workers)

        for key in checkpoint['state_dict'].keys(): 
            model_key = key.replace("module.", "")
            model_dict[model_key] = checkpoint['state_dict'][key] 

        model.load_state_dict(model_dict)
        start = time.time()
        class_activations = get_class_activations(model, val_loader)
        print("Time taken:", time.time() - start)
    
        epsilon = 1e-6

        act_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # k -> class -> Layer_k -> [[bn1], [bn2]...] where bni includes bottleneck values for each class [1000d mat]
        class_counter = {}
        # Layer_k = outer layer num, layer_v = dict of the form {class_i: {} ... }
        print("Calculating Activation values...")
        start = time.time()
        for layer_k, layer_v in class_activations.items():
            for class_k, class_v in class_activations[layer_k].items():
                for bottleneck_k, bottleneck_v in class_activations[layer_k][class_k].items():
                    for k in range(1, args.k+1):
                        top_k_activations_for_this_bottleneck, _ = torch.topk(class_activations[layer_k][class_k][bottleneck_k], k=k)
                        act_vals[k][class_k][layer_k].append(torch.mean(top_k_activations_for_this_bottleneck, dim=1).numpy())
    
        
        print("Activation values calculated...")
        print("Time taken: {}".format(time.time() - start))


        """
        Form a dataframe from the dict 
        Each column is length 1000 of the form lk_bi for k = layer num and i = bottleneck num 
        """

        print("Plotting...")

        for k, v in sorted(act_vals.items()):
            for class_k, class_v in sorted(act_vals[k].items()):
                df = pd.DataFrame()
                for layer, bns in sorted(act_vals[k][class_k].items()):
                    for i, bn_val in enumerate(bns): 
                        df["l{}_b{}".format(layer, i)] = bn_val 
                
                # df_dict[k][class_k] = df
                
                SAVE_DIR = EXP_DIR / 'k.{}'.format(k)
                SAVE_DIR.mkdir(parents=True, exist_ok=True)

                # Plot for all layers
                if args.plot_mat:
                    SAVE_DIR = EXP_DIR / 'k.{}'.format(k) / '{}.{}'.format(class_k, categories[class_k])
                    SAVE_DIR.mkdir(parents=True, exist_ok=True)
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
                    df_corr = df_sliced.corr()
                    # Take mean of abs correlation values between layer i and layer j 
                    corr_mat = np.abs(df_corr.loc[cols_i, cols_j].to_numpy())
                    # Save it in corr dict. Format:  Layer tuple -> Class Num -> list of mean values for each checkpoint 
                    corr_dict[k][(li, lj)][class_k].append(np.mean(corr_mat))
                    if args.plot_mat:
                        hm_sliced = sns.heatmap(df_corr, annot=True)
                        hm_sliced.set(title="CP {} Layers {} - {} for {}".format(cp, li, lj, categories[class_k]))
                        
                        SAVE_PATH = SAVE_DIR / 'cp{}_layers_{}_{}.jpg'.format(cp, li, lj)
                        
                        plt.savefig(SAVE_PATH)
                        plt.clf()
                    
                plt.close()
    
    # Save the corr_dict 
    with open(EXP_DIR / 'corr_dict', 'wb') as corr_file: 
        dill.dump(corr_dict, corr_file)
else: 
    with open(EXP_DIR / args.cd, 'rb') as cdict:
        corr_dict = dill.load(cdict)

sns.set_theme()

# Plot change of mean correlation across all classes
for k, v in corr_dict.items():
    SAVE_DIR = EXP_DIR / 'k.{}'.format(k)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for (li, lj), class_dict, in v.items():
        fig1, ax1 = plt.subplots()
        ax1.set_prop_cycle('color', colors)
        ax1.set_xlabel('Checkpoints')
        ax1.set_ylabel('Mean correlation')
        ax1.set_title(f'Layer {li}-{lj} k{k}')

        for class_k, means in class_dict.items(): 
            ax1.plot(checkpoints, means, label='Class {}'.format(class_k))
        
        fig1.savefig(SAVE_DIR / 'Layer_{}_{}_Mean_Corr.png'.format(li, lj))
        ax1.clear()

#  Error plot 
plt.clf()
for k, v in corr_dict.items(): 
    for (li, lj), class_dict, in v.items():
        check_means = defaultdict(list)
        final_means = []
        stds = []
        for class_k, means in class_dict.items(): 
            assert len(checkpoints) == len(means)
            for i, ci in enumerate(checkpoints): 
                check_means[ci].append(means[i])

        for ci, means in check_means.items(): 
            final_means.append(np.mean(means))
            stds.append(np.std(means))

        final_means, stds = np.array(final_means), np.array(stds)
        se = stds / np.sqrt(num_classes)
        confidence_intervals = 2 * se 

        plt.plot(checkpoints, final_means, 'b-', label='Mean of all classes')
        plt.fill_between(checkpoints, final_means - stds, final_means + stds, color='b', alpha=0.2)
        plt.xlabel('Checkpoints')
        plt.ylabel('Mean Correlation')
        plt.title(f'Layer {li}-{lj} k{k}')
        plt.legend()
        plt.savefig(EXP_DIR / f'Error_K{k}_Layer_{li}_{lj}.jpg')
        plt.clf()

        plt.plot(checkpoints, final_means, 'b-', label='Mean of all classes')
        plt.fill_between(checkpoints, final_means - confidence_intervals, final_means + confidence_intervals, color='b', alpha=0.2)
        plt.xlabel('Checkpoints')
        plt.ylabel('Mean Correlation')
        plt.title(f'Layer {li}-{lj} k{k}')
        plt.legend()
        plt.savefig(EXP_DIR / f'Confidence_Interval_K{k}_Layer_{li}_{lj}.jpg')
        plt.clf()



    
