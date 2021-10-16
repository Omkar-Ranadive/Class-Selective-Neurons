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
parser.add_argument("--exp_name", type=str, required=True)
args = parser.parse_args()


EXP_DIR = EXP_PATH / args.exp_name 


channels = {4: 256, 5: 512, 6: 1024, 7: 2048}

print(EXP_DIR)
checkpoint_files = sorted(glob.glob(os.path.join(EXP_DIR, "*.npy")))
# print("here:", checkpoint_files)
checkpoints = set()
for cp_file in checkpoint_files:
    cp = int(cp_file.rpartition('cp')[-1].split('_', 1)[0])
    # print(cp_file, cp)
    checkpoints.add(cp)

accuracy_type = ['t1', 't5']

# Set up the correct color palette - To avoid repeating on colors for plots with large number of lines
plt.style.use('fivethirtyeight')
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
            FILE_PATH = EXP_DIR / FILE_NAME
            acc = np.load(FILE_PATH)
            acc = (acc / np.max(acc)) * 100 

            # T5_FILE_NAME = r"t1_acc_cp{}_layer_{}.npy".format(cp, layer)
            # T5_FILE_PATH = os.path.join(EXP_DIR, T5_FILE_NAME)
            # t5_acc = np.load(T5_FILE_PATH)
            # t5_acc = t5_acc / t5_acc[0] * 100
            # print(t5_acc)
            # exit()

            ax.set_xlabel('Channels ablated')
            ax.set_ylabel('Accuracy')

            ax.plot(X, acc, label='CP {} Acc'.format(cp))
            # plt.plot(X, t5_acc, label='Top 5 CP {} Acc'.format(cp))
            ax.set_title('Layer {} Top {} Accuracies'.format(layer, t[-1]))
            ax.legend()
            # plt.savefig("./test_fig.png")
        fig.savefig(EXP_DIR / '{}_Top_{}_Normalized_Accuracy_Curves_cp{}_to_cp{}_layer_{}.png'.format(datetime.now().strftime('%m_%d_%Y-%H_%M_%S'), t[-1], min(checkpoints), max(checkpoints), layer))
        ax.clear()
        ax.set_prop_cycle('color', colors)


