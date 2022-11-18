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
import natsort 
import re 

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--num_workers", default=8, type=int)
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

channels = {4: 256, 5: 512, 6: 1024, 7: 2048}

checkpoints_to_load = [i for i in range(args.check_min, args.check_max+1)]

cs_for_every_cp = []

sns.set_theme(style='whitegrid')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.labelsize'] = 'medium'
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

# colors1 = sns.color_palette("Blues_d", 6)
# colors2 = sns.color_palette("Blues_d", 4)
cmap = plt.get_cmap('Blues')
# # cmap = sns.color_palette("Blues_d", as_cmap=True)
# # cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)

colors2 = cmap(np.linspace(0.45, 1.0, len(channels.keys()))) 

# colors1 = sns.color_palette("Blues_d", 4)
# colors2 = sns.color_palette("Blues_d", 6)


ax2.set_prop_cycle('color', colors2)


files = natsort.natsorted(os.listdir(DATA_DIR))
matches = ['.tar', 'full', '.log']
X_values = []
cp_previous = str(1) 
bcount = 1 
for f in files:
    if not any(s in f for s in matches):
        cs_dict_path = DATA_DIR / f
        class_selectivity = utils.load_file(cs_dict_path)
        batch_num = re.search('b\d+', f).group()
        cp = re.search('cp\d+', f).group()[2:]
        # cp = re.search('checkpoint_e\d+', f).group()[-1]
        # print(cp)
        if args.check_min <= int(cp) <= args.check_max:  
            if cp != cp_previous: 
                cp_previous = cp 
                bcount = 1 
            cs_for_every_cp.append(class_selectivity)
            X_values.append((cp +'b'+str(bcount)))
            bcount += 1
            
# X_values = range(0, len(cs_for_every_cp))

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

    # ax1.set_ylim([0.0, 0.7])
    # ax2.set_ylim([0.0, 0.7])
    # ax1.set_title(f'Module {l} Class Selectivity Index')
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Class Selectivity Index')
    # ax1.tick_params(axis='x', labelrotation = 0, labelsize='small')
    # # ax1.xaxis.set_major_locator(plt.MaxNLocator(10))


    # ax2.set_title(f'Class Selectivity Index across all modules')
    # ax2.set_xlabel('Epochs')
    # ax2.set_ylabel('Class Selectivity Index')
    # ax2.tick_params(axis='x', labelrotation = 0, labelsize='small')
    # # ax2.xaxis.set_major_locator(plt.MaxNLocator(10))

    module_level_means = []
    module_level_stds = []
    bkeys = cs_for_every_cp[0][l].keys()
    colors1 = cmap(np.linspace(0.45, 1.0, len(bkeys))) 
    ax1.set_prop_cycle('color', colors1)
    for b in bkeys:
        BOTTLENECK_LAYER_PATH = LAYER_PATH / "bottleneck_layer_{}".format(b)
        # os.makedirs(BOTTLENECK_LAYER_PATH, exist_ok=True)
      
        all_cs = []
        for i in range(c):
            CHANNEL_PATH = BOTTLENECK_LAYER_PATH / "channel_{}".format(i)
            # os.makedirs(CHANNEL_PATH, exist_ok=True)

            cs_for_channel = [cs_for_every_cp[x][l][b][i].item() for x in range(len(cs_for_every_cp))]

            all_cs.append(cs_for_channel)
        
        all_cs = np.array(all_cs) 
        means = np.mean(all_cs, axis=0) 
        module_level_means.append(means) 
        stds = np.std(all_cs, axis=0) 
        module_level_stds.append(stds)
        se = stds / np.sqrt(c)
        confidence_intervals = 2 * se        

        ax1.plot(X_values, means, label=f'Bottleneck Layer {b}')
        ax1.fill_between(X_values, means - confidence_intervals, means + confidence_intervals,  alpha=0.3)
        ax1.legend()
          
     
    module_level_means, module_level_stds = np.array(module_level_means), np.array(module_level_stds) 
    means, stds = np.mean(module_level_means, axis=0), np.std(module_level_stds, axis=0) 
    se = stds / np.sqrt(module_level_means.shape[0])
    confidence_intervals = 2 * se    

    ax2.plot(X_values, means, label=f'Module {l}')
    ax2.fill_between(X_values, means - confidence_intervals, means + confidence_intervals,  alpha=0.3)
    ax2.legend(loc='upper right')
          
    fig1.savefig(SAVE_DIR / f'l{l}_cp{args.check_min}_to_cp{args.check_max}.pdf', dpi=300, format='pdf')
    ax1.clear()
    # ax1.set_prop_cycle('color', colors2)


fig2.savefig(SAVE_DIR / f'All_Modules_cp{args.check_min}_to_cp{args.check_max}.pdf', dpi=300, format='pdf')
ax2.clear()





