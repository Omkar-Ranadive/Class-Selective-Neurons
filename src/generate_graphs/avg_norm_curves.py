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
parser.add_argument("--save_dir", default=None, type=str, required=True)
args = parser.parse_args()



CS_DIRS = ['ft1_ab', 'ft2_ab', 'ft3_ab', 'ft4_ab']
# CS_DIRS = ['ft1_ab']
RAN_DIRS = ['ft1_ab_ran', 'ft2_ab_ran', 'ft3_ab_ran', 'ft4_ab_ran']


SAVE_DIR = EXP_PATH / args.save_dir 
os.makedirs(SAVE_DIR, exist_ok=True)

channels = {4: 256, 5: 512, 6: 1024, 7: 2048}
accuracy_type = ['t1', 't5']
# Load all of the first 15 checkpoints as selectivity is more prominent in the earlier epochs 
checkpoints = [i for i in range(1, 16)]
# Then load remaining checkpoints with step of 10
checkpoints.extend(list(range(20, 91, 10)))


# Plot settings 
sns.set_theme(style='whitegrid')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.labelsize'] = 'medium'
number_of_plots = len(checkpoints)
# colors = sns.color_palette("hls", number_of_plots)
colors = sns.color_palette("rocket_r", number_of_plots)

# colors = sns.color_palette("flare")
# colors = sns.color_palette("crest")
# colors = sns.cubehelix_palette(start=.5, rot=-.75)

fig, ax = plt.subplots()
ax.set_prop_cycle('color', colors)
ax.tick_params(which='both', top=False, right=False)

fig2, ax2 = plt.subplots(figsize=(10, 6)) 
ax2.tick_params(which='both', top=False, right=False)

names = {'cs': ' Class selective ablations', 'ran': 'Random ablations'}
markers = {'cs': 'o', 'ran': '^'}

for t in accuracy_type:
    for layer in channels.keys():
        first_half = channels[layer]//2 
        X1 = list(range(0, first_half, 15))
        X2 = list(range(first_half, channels[layer]+1, 40)) 
        X = X1 + X2 
        for dirs, dir_type in [(CS_DIRS, 'cs'), (RAN_DIRS, 'ran')]:
            area_under_curve = []

            for cp in checkpoints:
                accuracies = []  
                for dir in dirs: 
                    FILE_NAME = f"{t}_acc_cp{cp}_layer_{layer}.npy"
                    FILE_PATH = EXP_PATH / dir / FILE_NAME
                    acc = np.load(FILE_PATH)
                    acc = (acc / np.max(acc)) * 100 
                    # print(f"Layer {layer}  X {len(X)}  CP {cp} acc {acc.shape}")
                    accuracies.append(acc)

                accuracies = np.mean(np.array(accuracies), axis=0)
                area_under_curve.append(sum(accuracies))
                # print(accuracies.shape)
                ax.set_xlabel('Channels ablated')
                ax.set_ylabel('Normalized Accuracy')
                ax.plot(X, accuracies)

    
            ax.set_title(f'Layer {layer} Top {t[-1]} Accuracies Type {dir_type}') 
            fig.savefig(SAVE_DIR / f'{dir_type}_Top_{t[-1]}_avg_norm_acc_layer_{layer}.pdf', format='pdf')
            ax.clear()
            ax.set_prop_cycle('color', colors)
            ax.tick_params(which='both', top=False, right=False)

            # AUC
            slope = (area_under_curve[-1] - area_under_curve[0]) / (checkpoints[-1] - checkpoints[0])
            ax2.plot(checkpoints, area_under_curve, label=f'{names[dir_type]}, slope={round(slope, 2)}', linewidth=2.5)
            ax2.legend()


        ax2.set_title(f'Area under the curve for layer {layer}')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Area under accuracy curve')
        fig2.savefig(SAVE_DIR / f'Top_{t[-1]}_AUC_layer_{layer}.pdf', format='pdf')
        ax2.clear()






