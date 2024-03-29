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

if args.data_dir is not None: 
    DATA_DIR = DATA_PATH / args.data_dir
else:
    DATA_DIR = DATA_PATH

channels = {4: 256, 5: 512, 6: 1024, 7: 2048}

checkpoints_to_load = [i for i in range(args.check_min, args.check_max)]

cs_for_every_cp = []

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()


for cp in checkpoints_to_load:
    cs_dict_path = DATA_DIR / 'cs_dict_{}_cp{}'.format(args.loader, cp)
    class_selectivity = utils.load_file(cs_dict_path)
    cs_for_every_cp.append(class_selectivity)

for l, c in channels.items():
    print("Plotting graphs for Layer {}...".format(l))
    LAYER_PATH = EXP_DIR / "layer_{}".format(l)
    os.makedirs(LAYER_PATH, exist_ok=True)
    number_of_plots = c
    colors = sns.color_palette("flare", number_of_plots)
    # Add alpha to the palette 
    alpha = 0.3
    colors = [(*color, alpha) for color in colors]

 
    for b in cs_for_every_cp[0][l].keys():
        BOTTLENECK_LAYER_PATH = LAYER_PATH / "bottleneck_layer_{}".format(b)
        os.makedirs(BOTTLENECK_LAYER_PATH, exist_ok=True)
        ax1.set_prop_cycle('color', colors)
        ax2.set_prop_cycle('color', colors)

        for i in range(c):
            CHANNEL_PATH = BOTTLENECK_LAYER_PATH / "channel_{}".format(i)
            os.makedirs(CHANNEL_PATH, exist_ok=True)

            cs_for_channel = np.load(CHANNEL_PATH / 'layer{}_bottleneck{}_channel{}_cp{}_to_cp{}.npy'.format(l, b, i, args.check_min, args.check_max))

            if (np.max(cs_for_channel) - np.min(cs_for_channel)) != 0: 
                cs_for_channel_norm = (cs_for_channel - np.min(cs_for_channel)) / (np.max(cs_for_channel) - np.min(cs_for_channel))
            else:
                cs_for_channel_norm = cs_for_channel

      
            # # Plot the un-normalized version first 
            ax1.set_xlabel('Checkpoints')
            ax1.set_ylabel('Class Selectivity Index')
            ax1.plot(checkpoints_to_load, cs_for_channel, label='CS for Channel {}'.format(i))
            ax1.set_title('Layer {} -- Bottleneck {}  -- Class Selectivity per Checkpoint'.format(l, b))

          
            # Plot the normalized version 
            ax2.set_xlabel('Checkpoints')
            ax2.set_ylabel('Class Selectivity Index')
            ax2.plot(checkpoints_to_load, cs_for_channel_norm, label='CS for Channel {}'.format(i))
            ax2.set_title('Layer {} -- Bottleneck {}  -- Class Selectivity per Checkpoint'.format(l, b))
        
        if args.save_dir is None: 
            SAVE_DIR = BOTTLENECK_LAYER_PATH 
        fig1.savefig(SAVE_DIR / '{}_l_{}_bn{}_cp{}_to_cp{}.png'.format(datetime.now().strftime('%m_%d_%Y-%H_%M_%S'), l, b, args.check_min, args.check_max))
        # fig2.savefig(SAVE_DIR / '{}_bn{}_cp{}_to_cp{}_norm.png'.format(datetime.now().strftime('%m_%d_%Y-%H_%M_%S'), b, args.check_min, args.check_max))

        ax1.clear()
        ax2.clear()



