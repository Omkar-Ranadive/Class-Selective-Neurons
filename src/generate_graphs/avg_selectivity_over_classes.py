import sys
from pathlib import Path 
src_path = Path(__file__).parent / '../'
sys.path.append(str(src_path))

import matplotlib.pyplot as plt 
import argparse 
import os
from datetime import datetime
import numpy as np
from constants import *
import utils
import seaborn as sns 
from matplotlib.ticker import AutoMinorLocator
import torch 


def get_selectivity(cs_dict_path, classes):

    class_selectivity = {
        4: {},
        5: {},
        6: {},
        7: {}
    }
    
    class_activations = utils.load_file(cs_dict_path)
    print(f"Calculating for epoch {cp}")
    for layer_k, layer_v in class_activations.items():
        # for class_k, class_v in class_activations[layer_k].items():
        # For a layer, the number of bottleneck layers will be the same 
        # So, just choose any class (in this case class 0) to get the index of bottleneck layers 
        for bottleneck_k, bottleneck_v in class_activations[layer_k][0].items():
            for index, class_k in enumerate(classes):
                if index > 0:
                    all_activations_for_this_bottleneck = np.concatenate((all_activations_for_this_bottleneck, class_activations[layer_k][class_k][bottleneck_k]), axis=0)
                else:
                    all_activations_for_this_bottleneck = class_activations[layer_k][class_k][bottleneck_k]
            
            all_activations_for_this_bottleneck = all_activations_for_this_bottleneck.T

            u_max = np.max(all_activations_for_this_bottleneck, axis=1)
            # print(all_activations_for_this_bottleneck.shape)
            # avg_act = torch.mean(all_activations_for_this_bottleneck, dim=0).numpy()
            # print(avg_act.shape)
            u_sum = np.sum(all_activations_for_this_bottleneck, axis=1)
            u_minus_max = (u_sum - u_max) / (all_activations_for_this_bottleneck.shape[1] - 1)

            selectivity = (u_max - u_minus_max) / (u_max + u_minus_max + EPSILON)
            
            class_selectivity[layer_k].update({bottleneck_k: selectivity})
    
    return class_selectivity



parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--loader", default='val', type=str)
parser.add_argument("--check_min", required=True, type=int)
parser.add_argument("--check_max", required=True, type=int)
parser.add_argument("--save_dir", default=None, type=str)
parser.add_argument("--arc", default='resnet50', type=str)
parser.add_argument("--bins", default=6, type=int)
parser.add_argument("--format", type=str, default='pdf')
parser.add_argument("--dpi", default=300, type=int)
parser.add_argument("--cmin", type=int, required=True) 
parser.add_argument("--cmax", type=int, required=True) 

# Refer class names here: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

args = parser.parse_args()


EXP_DIR = EXP_PATH / args.exp_name 
os.makedirs(EXP_DIR, exist_ok=True)


if args.save_dir is not None: 
    SAVE_DIR = EXP_PATH / args.save_dir 
    os.makedirs(SAVE_DIR, exist_ok=True)
else: 
    SAVE_DIR = EXP_DIR


if args.arc == 'resnet50':
    channels = {4: 256, 5: 512, 6: 1024, 7: 2048}
elif args.arc == 'resnet18' or args.arc == 'resnet34':
    channels = {4: 64, 5: 128, 6: 256, 7: 512}

dirs_og = ['rn50_1', 'rn50_2', 'rn50_3', 'rn50_4', 'rn50_5']
dirs_e0 = ['rn50_a20e0_1', 'rn50_a20e0_2', 'rn50_a20e0_3', 'rn50_a20e0_4', 'rn50_a20e0_5']
dirs_e5 = ['rn50_a20e5_1', 'rn50_a20e5_2', 'rn50_a20e5_3', 'rn50_a20e5_4', 'rn50_a20e5_5']
names = ['Original unregularized model', 'Regularized from epoch 0 onward', 'Regularized from epoch 5 onward']

checkpoints_to_load = [i for i in range(args.check_min, args.check_max+1)]
classes = list(range(args.cmin, args.cmax+1))

cs_for_every_cp = []

sns.set_theme(style='whitegrid')
plt.rcParams['savefig.dpi'] = args.dpi 
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()



for dir_index, dirs in enumerate([dirs_og]):
    if dir_index == 0: 
        cmap = plt.get_cmap('Greens') 
    elif dir_index == 1: 
        cmap = plt.get_cmap('OrRd')
    elif dir_index == 2:     
        cmap = plt.get_cmap('Blues')

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Class Selectivity Index')
    ax2.xaxis.set_major_locator(plt.MaxNLocator(nbins=args.bins, integer=True))
    ax2.set_title(f'Selectivity Index for Classes: {args.cmin}-{args.cmax}')
    ax2.set_ylim(top=0.75)

    ax2.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax2.tick_params(axis='x', which='minor', length=3, color='r', direction='out')
    ax2.tick_params(which='both', top=False, right=False, bottom=True)
    
    colors2 = cmap(np.linspace(0.45, 1.0, len(channels.keys()))) 
    ax2.set_prop_cycle('color', colors2)

    cs_dirs = []
    for dir in dirs:  
        DATA_DIR = DATA_PATH / dir 
        cs_for_every_cp = []
        for cp in checkpoints_to_load:
            cs_dict_path = DATA_DIR / 'cs_dict_{}_cp{}_full'.format(args.loader, cp)
            class_selectivity = get_selectivity(cs_dict_path, classes)
            cs_for_every_cp.append(class_selectivity)

        cs_dirs.append(cs_for_every_cp)
    
    for l, c in channels.items():
        print("Plotting graphs for Layer {}...".format(l))
        LAYER_PATH = EXP_DIR / "layer_{}".format(l)
        # os.makedirs(LAYER_PATH, exist_ok=True)
        number_of_plots = c
        
        ax1.set_title(f'Module {l} Class Selectivity Index')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Class Selectivity Index')
        ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=args.bins, integer=True))
        ax1.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax1.tick_params(axis='x', which='minor', length=3, color='r', direction='out')
        ax1.tick_params(which='both', top=False, right=False, bottom=True)

        
        module_level_means = []
        module_level_stds = []

        bkeys = cs_dirs[0][0][l].keys()
        colors1 = cmap(np.linspace(0.45, 1.0, len(bkeys))) 
        ax1.set_prop_cycle('color', colors1)
        for b in bkeys:
            all_dirs_cs = []
            for cs_for_every_cp in cs_dirs: 
                BOTTLENECK_LAYER_PATH = LAYER_PATH / "bottleneck_layer_{}".format(b)
                all_cs = []
                for i in range(c):
                    CHANNEL_PATH = BOTTLENECK_LAYER_PATH / "channel_{}".format(i)
                    # os.makedirs(CHANNEL_PATH, exist_ok=True)

                    # cs_for_channel = np.load(CHANNEL_PATH / 'layer{}_bottleneck{}_channel{}_cp{}_to_cp{}.npy'.format(l, b, i, args.check_min, args.check_max))
                    cs_for_channel = [cs_for_every_cp[x][l][b][i].item() for x in range(len(cs_for_every_cp))]

                    all_cs.append(cs_for_channel)

                all_dirs_cs.append(all_cs)
            
            all_dirs_cs = np.array(all_dirs_cs)
            means = np.mean(all_dirs_cs, axis=(0, 1)) 
            module_level_means.append(means) 
            stds = np.std(all_dirs_cs, axis=(0, 1)) 
            module_level_stds.append(stds)
            se = stds / np.sqrt(c)
            confidence_intervals = 2 * se        
            
            ax1.plot(checkpoints_to_load, means, label=f'Bottleneck Layer {b}')
            ax1.fill_between(checkpoints_to_load, means - confidence_intervals, means + confidence_intervals,  alpha=0.3)
            ax1.legend()
            

        module_level_means, module_level_stds = np.array(module_level_means), np.array(module_level_stds) 
        means, stds = np.mean(module_level_means, axis=0), np.std(module_level_stds, axis=0) 
        se = stds / np.sqrt(module_level_means.shape[0])
        confidence_intervals = 2 * se    

        ax2.plot(checkpoints_to_load, means, label=f'Module {l}')
        ax2.fill_between(checkpoints_to_load, means - confidence_intervals, means + confidence_intervals,  alpha=0.3)
        ax2.legend(loc='upper right')

        if dir_index == 1: 
            ax2.vlines(0, 0, 0.75, color='k', alpha=0.9, linestyles='dashed')
        elif dir_index == 2: 
            ax2.vlines(5, 0, 0.75, color='k', alpha=0.7, linestyles='dashed')
            
        fig1.savefig(SAVE_DIR / f'{names[dir_index]}_l{l}_cp{args.check_min}_to_cp{args.check_max}_c{args.cmin}_c{args.cmax}.{args.format}', 
                     format=args.format)

        ax1.clear()

    fig2.savefig(SAVE_DIR / f'{names[dir_index]}_All_Modules_cp{args.check_min}_to_cp{args.check_max}c{args.cmin}_c{args.cmax}.{args.format}', 
                 format=args.format)

    ax2.clear()





