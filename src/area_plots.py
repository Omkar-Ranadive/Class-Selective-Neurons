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



EXP_DIR = EXP_PATH / args.exp_name_cs
RAN_DIR = EXP_PATH / args.exp_name_ran

if args.save_dir is not None: 
    SAVE_DIR = EXP_PATH / args.save_dir 
else: 
    SAVE_DIR = EXP_DIR 

os.makedirs(SAVE_DIR, exist_ok=True)


channels = {4: 256, 5: 512, 6: 1024, 7: 2048}
accuracy_type = ['t1', 't5']


for t in accuracy_type:
    for layer in channels.keys():
        X = list(range(0, channels[layer]+1, 10)) # Stepping the channels to speed it up 

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Layer {} Top {} Accuracies'.format(layer, t[-1]))
        ax.set_xlabel('Checkpoints')
        ax.set_ylabel('Area under Accuracy Curve')

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
            for cp in checkpoints:
                FILE_NAME = "{}_acc_cp{}_layer_{}.npy".format(t, cp, layer)
                FILE_PATH = CUR_DIR / FILE_NAME
                acc = np.load(FILE_PATH)
                acc = (acc / np.max(acc)) * 100 

                area_under_curve.append(sum(acc))
            
            ax.plot(list(checkpoints), area_under_curve, label='Type {}'.format(cur_type))
            ax.legend() 
            
        fig.savefig(SAVE_DIR / '{}_Top_{}_Normalized_Area_Under_Accuracy_Curves_cp{}_to_cp{}_layer_{}.png'.format(datetime.now().strftime('%m_%d_%Y-%H_%M_%S'), t[-1], min(checkpoints), max(checkpoints), layer))
        ax.clear()
