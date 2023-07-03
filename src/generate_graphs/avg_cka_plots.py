import sys
from pathlib import Path 
src_path = Path(__file__).parent / '../'
sys.path.append(str(src_path))
import torch
from torch import nn
import argparse 
from constants import *
import os 
from itertools import combinations_with_replacement 
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator, LinearLocator
import cycler 
import numpy as np 
from natsort import natsorted


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--check_min", type=int, default=0)
parser.add_argument("--check_max", type=int, default=20)

args = parser.parse_args()

# Plot params 
sns.set_theme(style='whitegrid')
plt.rcParams['savefig.dpi'] = 300
fig1, ax1 = plt.subplots()
ax1.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
ax1.tick_params(which='both', left=False, bottom=False, top=False, right=False)

plt.rcParams['axes.titlesize'] = 'large'
plt.rcParams['axes.labelsize'] = 'medium'
# plt.rcParams['axes.titlesize'] = 18
# plt.rcParams['axes.labelsize'] = 15
plt.rcParams['lines.linewidth'] = 2

EXP_DIR = EXP_PATH / args.exp_name 
os.makedirs(EXP_DIR, exist_ok=True)


names = {0: 'Module 4', 1: 'Module 5', 2: 'Module 6', 3: 'Module 7', 4: 'FC Layer',
         'og': 'Unregularized model - Unregularized model', 'e0': 'Unregularized model - Reg from epoch 0', 
         'e5': 'Unregularized model - Reg from epoch 5'
         }


dirs_og = ['rn50_1_2', 'rn50_1_3', 'rn50_1_4', 'rn50_1_5', 'rn50_2_3', 
           'rn50_2_4', 'rn50_2_5', 'rn50_3_4', 'rn50_3_5', 'rn50_4_5']

dirs_e0 = ['rn50_1_e01', 'rn50_1_e02', 'rn50_1_e03', 'rn50_1_e04', 'rn50_1_e05', 
           'rn50_2_e01', 'rn50_2_e02', 'rn50_3_e03', 'rn50_4_e04', 'rn50_5_e05']

dirs_e5 = ['rn50_1_e51', 'rn50_1_e52', 'rn50_1_e53', 'rn50_1_e54', 'rn50_1_e55', 
           'rn50_2_e51', 'rn50_2_e52', 'rn50_3_e53', 'rn50_4_e54', 'rn50_5_e55']

all_dirs = [(dirs_og, 'og'), (dirs_e0, 'e0')]

layers = list(combinations_with_replacement(range(0, 5), 2)) 
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors[0], colors[2] = colors[2], colors[0]  


for li, lj in layers: 
    for index, (dirs, dir_type) in enumerate(all_dirs): 
        all_scores = []
        for dir in dirs:
            scores = [] 
            DATA_DIR = EXP_PATH / dir 
            for f in natsorted(os.listdir(DATA_DIR)): 
                if f.endswith('.pt'): 
                    cka_mat = torch.load(DATA_DIR / f).numpy() 
                    mean_val = (cka_mat[li, lj] + cka_mat[lj, li])/2 
                    scores.append(round(mean_val, 4))
            
            all_scores.append(scores)
        
        all_scores = np.array(all_scores) 
        avg_scores = np.mean(all_scores, axis=0)
        se = np.std(all_scores, axis=0) / np.sqrt(len(dirs))
        confidence_intervals = 2 * se   
        ax1.plot(range(args.check_min, args.check_max+1), avg_scores, label=f'{names[dir_type]}', alpha=0.8, c=colors[index], lw=2)
        # ax1.fill_between(range(args.check_min, args.check_max+1), avg_scores - confidence_intervals, avg_scores + confidence_intervals, 
        #              alpha=0.3, color=colors[index])

    ax1.set_title(f'CKA between {names[li]} and {names[lj]}')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax1.legend()
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('CKA Similarity')
    ax1.tick_params(which='both', left=False, bottom=False, top=False, right=False)
    fig1.savefig(EXP_DIR / f'cka_{names[li]}_{names[lj]}.pdf', format='pdf')
    ax1.clear()
 








