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
import re 
from itertools import product 

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name_cs", type=str, required=True)
parser.add_argument("--exp_name_ran", type=str, required=True)
parser.add_argument("--save_dir", default=None, type=str)
args = parser.parse_args()


sns.set_theme()
EXP_DIR = EXP_PATH / args.exp_name_cs
RAN_DIR = EXP_PATH / args.exp_name_ran

if args.save_dir is not None: 
    SAVE_DIR = EXP_PATH / args.save_dir 
else: 
    SAVE_DIR = EXP_DIR 

os.makedirs(SAVE_DIR, exist_ok=True)


channels = {4: 256, 5: 512, 6: 1024, 7: 2048}
accuracy_type = ['t1', 't5']

colors = {'cs': 'b', 'ran':'r'}

for t in accuracy_type:
    for layer in channels.keys():
        X = list(range(0, channels[layer]+1, 10)) # Stepping the channels to speed it up 

        fig, ax = plt.subplots(figsize=(10, 6))
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        ax.set_title('Layer {} Top {} Accuracies'.format(layer, t[-1]))
        ax.set_xlabel('Checkpoints')
        ax.set_ylabel('Area under Accuracy Curve')

        ax2.set_title('Layer {} Top {} Accuracies'.format(layer, t[-1]))
        ax2.set_xlabel('Checkpoints')
        ax2.set_ylabel('Mean accuracy')

        for CUR_DIR, cur_type in [(EXP_DIR, 'cs'), (RAN_DIR, 'ran')]: 
            checkpoint_files = sorted(glob.glob(os.path.join(CUR_DIR, "*.npy")))
            # print("here:", checkpoint_files)
            checkpoints = set()
            for cp_file in checkpoint_files:
                cp = int(cp_file.rpartition('cp')[-1].split('_', 1)[0])
                # Ignore cp 0, as it doesn't add much to the curve 
                if cp != 0:
                    checkpoints.add(cp)

            area_under_curve = []
            final_means = []
            stds = []
            for cp in checkpoints:
                FILE_NAME = "{}_acc_cp{}_layer_{}.npy".format(t, cp, layer)
                FILE_PATH = CUR_DIR / FILE_NAME
                acc = np.load(FILE_PATH)

                acc = (acc / np.max(acc)) * 100 

                area_under_curve.append(sum(acc))
                final_means.append(np.mean(acc))
                stds.append(np.std(acc))

            ax.plot(list(checkpoints), area_under_curve, label='Type {}'.format(cur_type))
            ax.legend() 
            final_means, stds = np.array(final_means), np.array(stds)
            
            se = stds / np.sqrt(len(acc))
            confidence_intervals = 2 * se 

            ax2.plot(list(checkpoints), final_means, colors[cur_type]+'-', label=f'Mean of accuracies for {cur_type}')
            ax2.fill_between(list(checkpoints), final_means - confidence_intervals, final_means + confidence_intervals, color=colors[cur_type], alpha=0.2)
            ax2.legend()

            
        fig.savefig(SAVE_DIR / 'Top_{}_Normalized_Area_Under_Accuracy_Curves_cp{}_to_cp{}_layer_{}.png'.format(t[-1], min(checkpoints), max(checkpoints), layer))
        fig2.savefig(SAVE_DIR / 'CI_Top_{}_Normalized_Area_Under_Accuracy_Curves_cp{}_to_cp{}_layer_{}.png'.format(t[-1], min(checkpoints), max(checkpoints), layer))

        ax.clear()
        ax2.clear()
