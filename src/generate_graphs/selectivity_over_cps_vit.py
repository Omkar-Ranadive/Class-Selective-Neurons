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

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--loader", default='val', type=str)
parser.add_argument("--check_min", required=True, type=int)
parser.add_argument("--check_max", required=True, type=int)
parser.add_argument("--save_dir", default=None, type=str)


args = parser.parse_args()


EXP_DIR = EXP_PATH / args.exp_name 
os.makedirs(EXP_DIR, exist_ok=True)


if args.save_dir is not None: 
    SAVE_DIR = EXP_PATH / args.save_dir 
    os.makedirs(SAVE_DIR, exist_ok=True)
else: 
    SAVE_DIR = EXP_DIR

if args.data_dir is not None: 
    DATA_DIR = DATA_PATH / args.data_dir
else:
    DATA_DIR = DATA_PATH

channels = {4: 197, 5: 197, 6: 197, 7: 197}

checkpoints_to_load = [i for i in range(args.check_min, args.check_max+1)]

cs_for_every_cp = []

sns.set_theme()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

ax1.clear()
ax2.clear()

for cp in checkpoints_to_load:
    cs_dict_path = DATA_DIR / 'cs_dict_{}_cp{}'.format(args.loader, cp)
    class_selectivity = utils.load_file(cs_dict_path)
    cs_for_every_cp.append(class_selectivity)

for l, c in channels.items():
    print("Plotting graphs for Layer {}...".format(l))
    LAYER_PATH = EXP_DIR / "layer_{}".format(l)
    # os.makedirs(LAYER_PATH, exist_ok=True)
    number_of_plots = c
    
    ax1.set_title(f'Module {l} Class Selectivity Index')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Class Selectivity Index')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax1.tick_params(axis='x', which='minor', length=3, color='r', direction='out')
    ax1.tick_params(which='both', top=False, right=False)

    ax2.set_title(f'Class Selectivity Index across all modules')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Class Selectivity Index')
    ax2.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax2.tick_params(axis='x', which='minor', length=3, color='r', direction='out')
    ax2.tick_params(which='both', top=False, right=False)




    module_level_means = []
    module_level_stds = []
    for b in cs_for_every_cp[0][l].keys():
        BOTTLENECK_LAYER_PATH = LAYER_PATH / "bottleneck_layer_{}".format(b)
        # os.makedirs(BOTTLENECK_LAYER_PATH, exist_ok=True)
      
        all_cs = []
        for i in range(c):
            CHANNEL_PATH = BOTTLENECK_LAYER_PATH / "channel_{}".format(i)
            # os.makedirs(CHANNEL_PATH, exist_ok=True)

            # cs_for_channel = np.load(CHANNEL_PATH / 'layer{}_bottleneck{}_channel{}_cp{}_to_cp{}.npy'.format(l, b, i, args.check_min, args.check_max))
            cs_for_channel = [cs_for_every_cp[x][l][b][i].item() for x in range(len(cs_for_every_cp))]

            all_cs.append(cs_for_channel)
        
        all_cs = np.array(all_cs) 
        means = np.mean(all_cs, axis=0) 
        module_level_means.append(means) 
        stds = np.std(all_cs, axis=0) 
        module_level_stds.append(stds)
        se = stds / np.sqrt(c)
        confidence_intervals = 2 * se        

        print("B", checkpoints_to_load, means)
        ax1.plot(checkpoints_to_load, means, label=f'Bottleneck Layer {b}', marker='o', markersize=5)
        ax1.fill_between(checkpoints_to_load, means - confidence_intervals, means + confidence_intervals,  alpha=0.3)
        ax1.legend()
          
     
    module_level_means, module_level_stds = np.array(module_level_means), np.array(module_level_stds) 
    means, stds = np.mean(module_level_means, axis=0), np.std(module_level_stds, axis=0) 
    se = stds / np.sqrt(module_level_means.shape[0])
    confidence_intervals = 2 * se    
    print("M", checkpoints_to_load, means)
    ax2.plot(checkpoints_to_load, means, label=f'Module {l}', marker='o', markersize=5)
    ax2.fill_between(checkpoints_to_load, means - confidence_intervals, means + confidence_intervals,  alpha=0.3)
    ax2.legend(loc='upper right')
          
    fig1.savefig(SAVE_DIR / f'{args.data_dir}_l{l}_cp{args.check_min}_to_cp{args.check_max}.png')

    ax1.clear()

fig2.savefig(SAVE_DIR / f'{args.data_dir}_All_Modules_cp{args.check_min}_to_cp{args.check_max}.png')

ax2.clear()





