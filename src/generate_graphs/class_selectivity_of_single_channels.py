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


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--loader", default='val', type=str)
parser.add_argument("--check_min", required=True, type=int)
parser.add_argument("--check_max", required=True, type=int)
parser.add_argument("--plot", default='true', type=str)
args = parser.parse_args()


args.plot = True if args.plot == 'true' else False 

EXP_DIR = EXP_PATH / args.exp_name 
os.makedirs(EXP_DIR, exist_ok=True)

channels = {4: 256, 5: 512, 6: 1024, 7: 2048}

checkpoints_to_load = [i for i in range(args.check_min, args.check_max)]

cs_for_every_cp = []

for cp in checkpoints_to_load:
    cs_dict_path = DATA_PATH / 'cs_dict_{}_cp{}'.format(args.loader, cp)
    class_selectivity = utils.load_file(cs_dict_path)
    cs_for_every_cp.append(class_selectivity)

for l, c in channels.items():
    print("Plotting graphs for Layer {}...".format(l))
    LAYER_PATH = EXP_DIR / "layer_{}".format(l)
    os.makedirs(LAYER_PATH, exist_ok=True)
    for b in cs_for_every_cp[0][l].keys():
        BOTTLENECK_LAYER_PATH = LAYER_PATH / "bottleneck_layer_{}".format(b)
        os.makedirs(BOTTLENECK_LAYER_PATH, exist_ok=True)
        for i in range(c):
            CHANNEL_PATH = BOTTLENECK_LAYER_PATH / "channel_{}".format(i)
            os.makedirs(CHANNEL_PATH, exist_ok=True)

            cs_for_channel = [cs_for_every_cp[x][l][b][i].item() for x in range(len(cs_for_every_cp))]

            if args.plot: 
                plt.xlabel('Checkpoints')
                plt.ylabel('Class Selectivity Index')
                plt.plot(checkpoints_to_load, cs_for_channel, label='CS for Channel {}'.format(i))
                plt.title('Layer {} -- Bottleneck {} -- Channel {} -- Class Selectivity per Checkpoint'.format(l, b, i))
                plt.savefig(CHANNEL_PATH / '{}_channel_{}_cp{}_to_cp{}.png'.format(datetime.now().strftime('%m_%d_%Y-%H_%M_%S'), i, args.check_min, args.check_max))
                plt.clf()

            np.save(CHANNEL_PATH / 'layer{}_bottleneck{}_channel{}_cp{}_to_cp{}'.format(l, b, i, args.check_min, args.check_max), cs_for_channel)