import sys
from pathlib import Path 
src_path = Path(__file__).parent / '../'
sys.path.append(str(src_path))

import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import os
from datetime import datetime
from constants import EXP_PATH
import seaborn as sns



parser = argparse.ArgumentParser()
parser.add_argument("--exp_name_cs", type=str, required=True)
parser.add_argument("--exp_name_ran", type=str, required=True)
parser.add_argument("--save_dir", default=None, type=str)
args = parser.parse_args()



EXP_DIR = EXP_PATH / args.exp_name_cs
RAN_DIR = EXP_PATH / args.exp_name_ran

if args.save_dir is not None: 
    SAVE_DIR = EXP_PATH / args.save_dir 
else: 
    SAVE_DIR = EXP_DIR 

os.makedirs(SAVE_DIR, exist_ok=True)


channels = {4: 256, 5: 512, 6: 1024, 7: 2048}

for CUR_DIR, cur_type in [(EXP_DIR, 'cs'), (RAN_DIR, 'ran')]: 
    checkpoint_files = sorted(glob.glob(os.path.join(CUR_DIR, "*.npy")))
    # print("here:", checkpoint_files)
    checkpoints = set()
    for cp_file in checkpoint_files:
        cp = int(cp_file.rpartition('cp')[-1].split('_', 1)[0])
        # print(cp_file, cp)
        # Ignore cp 0, as it doesn't add much to the curve 
        if cp != 0:
            checkpoints.add(cp)

    accuracy_type = ['t1', 't5']

    # Set up the correct color palette - To avoid repeating on colors for plots with large number of lines
    # plt.style.use('fivethirtyeight')
    number_of_plots = len(checkpoints)
    colors = sns.color_palette("hls", number_of_plots)
    # colors = sns.color_palette("flare", as_cmap=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_prop_cycle('color', colors)



    for t in accuracy_type:
        for layer in channels.keys():
            X = list(range(0, channels[layer]+1, 10)) # Stepping the channels to speed it up 
            for cp in checkpoints:
                FILE_NAME = "{}_acc_cp{}_layer_{}.npy".format(t, cp, layer)
                FILE_PATH = CUR_DIR / FILE_NAME
                acc = np.load(FILE_PATH)
                acc = (acc / np.max(acc)) * 100 

                ax.set_xlabel('Channels ablated')
                ax.set_ylabel('Normalized Accuracy')

                ax.plot(X, acc, label='CP {} Acc'.format(cp))
                # plt.plot(X, t5_acc, label='Top 5 CP {} Acc'.format(cp))
                ax.set_title('Layer {} Top {} Accuracies Type {}'.format(layer, t[-1], cur_type))
                # ax.legend() 
                # plt.savefig("./test_fig.png")
            
            fig.savefig(SAVE_DIR / '{}_{}_Top_{}_Normalized_Accuracy_Curves_cp{}_to_cp{}_layer_{}.png'.format(datetime.now().strftime('%m_%d_%Y-%H_%M_%S'), cur_type, t[-1], min(checkpoints), max(checkpoints), layer))
            ax.clear()
            ax.set_prop_cycle('color', colors)


