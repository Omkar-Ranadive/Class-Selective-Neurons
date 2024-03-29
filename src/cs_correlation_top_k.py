from constants import DATA_PATH, EXP_PATH, IMGNET_PATH, IMGNET_CLASSES
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
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--k", default=3, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--plot_mat", action='store_true')
parser.add_argument("--cd", default=None, type=str, help="Can provide the correlation dictionary directly if available")
parser.add_argument("--ran", action='store_true', help="If true, select random k neurons instead of top k class selective ones")
args = parser.parse_args()


EXP_DIR = EXP_PATH / args.exp_name
os.makedirs(EXP_DIR, exist_ok=True)

data_dir = DATA_PATH / args.data_dir

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
                    # Output after reshaping - [batch_size, bottlneck_size], example (512, 256)
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
            
            # For final layer 9 (linear output)
            if isinstance(layer, torch.nn.modules.linear.Linear):
                activations = input_batch.detach().clone()
                num = 0  # For layer 9 aka the Linear layer, there is no bottleneck layer. For simplicity, use bottleneck num as 0 

                for i, activation in enumerate(activations): 
                    activation = torch.unsqueeze(activation, dim=0)

                    if target[i].item() not in class_activations[index]:
                        class_activations[index].update({target[i].item(): {}})  # ex: {layer_3: {class_0: {} } }
                    
                    if num in class_activations[index][target[i].item()]:
                        # class_activations[index][target[i].item()][num] += activation.cpu()
                        class_activations[index][target[i].item()][num] = torch.cat((class_activations[index][target[i].item()][num], activation.cpu()), dim=0)
                    else:
                        class_activations[index][target[i].item()].update({num: activation.cpu()})  # ex: {layer_3: {class_0: {bottleneck_0: activation} } }



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
        7: {}, 
        9: {}
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

# checkpoints = range(1, 90, 4)  # Step by 4 
# checkpoints = [0] + list(checkpoints)
# checkpoints = [1, 45, 89]
checkpoints = [i for i in range(0, 21)]


if not args.cd: 
    corr_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for cp in checkpoints:
        logger.info("-"*20)
        logger.info("Checkpoint {}".format(cp))
        # class_activations = utils.load_file(DATA_PATH / 'cs_dict_val_cp{}_full'.format(cp))
        # checkpoint = torch.load(DATA_PATH / 'model_checkpoints' / 'CHECKPOINTS' / 'EXPE1' / 'checkpoint_epoch{}.pth.tar'.format(cp))
        checkpoint = torch.load(data_dir / f'checkpoint_e{cp}.pth.tar')

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
                        if not args.ran: 
                            # Size of this is (50, k) where 50 is total num of examples per class in validation set  
                            k_activations_for_this_bottleneck, _ = torch.topk(class_activations[layer_k][class_k][bottleneck_k], k=k)
                        else: 
                            indices = np.random.randint(low=0, high=class_activations[layer_k][class_k][bottleneck_k].shape[1], size=k)
                            k_activations_for_this_bottleneck = class_activations[layer_k][class_k][bottleneck_k][:, indices]

                        # The activations are then averaged across k. So for k=3, (50, 3) becomes (50, ).
                        # So, there will be activation values per class example for each bottleneck layer. 
                        # Example -  for layer 4, if bottleneck layers = 3, then we have a vector of (50, ) for 3 bottleneck layers.  
                        act_vals[k][class_k][layer_k].append(torch.mean(k_activations_for_this_bottleneck, dim=1).numpy())
                        # print("Shape after taking mean: ",  act_vals[k][class_k][layer_k][-1].shape)

            # print(f"Act vals length for layer{layer_k} and k{k}: {len(act_vals[k][class_k][layer_k])}")
        
        print("Activation values calculated...")
        print("Time taken: {}".format(time.time() - start))

        print("Plotting...")

        for k, v in sorted(act_vals.items()):
            for class_k, class_v in sorted(act_vals[k].items()):
                df = pd.DataFrame()
                for layer, bns in sorted(act_vals[k][class_k].items()):
                    for i, bn_val in enumerate(bns): 
                        # print(f"Class {class_k} Layer {layer} Bottleneck {i} bnval {bn_val.shape}")
                        df["l{}_b{}".format(layer, i)] = bn_val 
                
                # df_dict[k][class_k] = df
                
                SAVE_DIR = EXP_DIR / 'k.{}'.format(k)
                SAVE_DIR.mkdir(parents=True, exist_ok=True)

                # Plot for all layers
                if args.plot_mat:
                    SAVE_DIR = EXP_DIR / 'k.{}'.format(k) / '{}.{}'.format(class_k, categories[class_k])
                    SAVE_DIR.mkdir(parents=True, exist_ok=True)
                    hm = sns.heatmap(df.corr())
                    hm.set(title="CP {} all modules for {}".format(cp, categories[class_k]))
                    
                    SAVE_PATH = SAVE_DIR / 'cp{}_all_modules.jpg'.format(cp)
                    
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
                    # print("*"*20)
                    # print(f"Shape of dataframe for layers {li} - {lj}, {df_sliced.shape}")
                    # print(df_sliced)
                    # Plot for the sliced frame 
                    df_corr = df_sliced.corr()
                    # print("*"*20)
                    # print("Correlation between the columns: ")
                    # print(df_corr.shape)
                    # print(df_corr)
                    # Take mean of abs correlation values between layer i and layer j 
                    corr_mat = np.abs(df_corr.loc[cols_i, cols_j].to_numpy())
                    # Save it in corr dict. Format:  Layer tuple -> Class Num -> list of mean values for each checkpoint 
                    # Take nan mean as if std = 0, then corr for that pair will be Nan 
                    corr_dict[k][(li, lj)][class_k].append(np.nanmean(corr_mat))
                    # print("Final class mean", corr_dict[k][(li, lj)][class_k])
                    if args.plot_mat:
                        hm_sliced = sns.heatmap(df_corr, annot=True)
                        hm_sliced.set(title="CP {} Modules {} - {} for {}".format(cp, li, lj, categories[class_k]))
                        
                        SAVE_PATH = SAVE_DIR / 'cp{}_modules_{}_{}.jpg'.format(cp, li, lj)
                        
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
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Mean correlation')
        ax1.set_title(f'Module {li}-{lj} k{k}')

        for class_k, means in class_dict.items(): 
            ax1.plot(checkpoints, means, label='Class {}'.format(class_k))
        
        fig1.savefig(SAVE_DIR / 'Module_{}_{}_Mean_Corr.png'.format(li, lj))
        ax1.clear()
        plt.close(fig1)

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
        plt.xlabel('Epochs')
        plt.ylabel('Mean Correlation')
        if not args.ran: 
            plt.title(f'Correlation between class selective neurons (k={k}) of module {li}-{lj}')
        else: 
            plt.title(f'Correlation between random neurons (k={k}) of module {li}-{lj}')


        plt.legend()
        plt.savefig(EXP_DIR / f'Error_K{k}_Module_{li}_{lj}.jpg')
        plt.clf()

        plt.plot(checkpoints, final_means, 'b-', label='Mean of all classes')
        plt.fill_between(checkpoints, final_means - confidence_intervals, final_means + confidence_intervals, color='b', alpha=0.2)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Correlation')
        if not args.ran: 
            plt.title(f'Correlation between class selective neurons (k={k}) of module {li}-{lj}')
        else: 
            plt.title(f'Correlation between random neurons (k={k}) of module {li}-{lj}')
        plt.legend()
        plt.savefig(EXP_DIR / f'Confidence_Interval_K{k}_Module_{li}_{lj}.png')
        plt.clf()



    
