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

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--m1", default=None, type=str)
parser.add_argument("--m2", default=None, type=str)
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
names = {0: 'Module 4', 1: 'Module 5', 2: 'Module 6', 3: 'Module 7', 4: 'Fully Connected Layer',
         'ft_5':'Unregularized model', 'ft_5_og': 'Original unregularized model', 
         'ft_7': 'Unregularized model2',
         'ft_5_regN5': 'Reg from epoch 0', 'ft_5_regN5_2': 'Reg from epoch 5'}

DATA_DIR = EXP_PATH / args.data_dir

if args.m1 and args.m2: 
    models = [(args.m1, args.m2)]
else: 
    models = [('ft_5', 'ft_5_regN5'), ('ft_5', 'ft_5_regN5_2'), ('ft_5', 'ft_7')]

layers = list(combinations_with_replacement(range(0, 5), 2)) 

for li, lj in layers: 
    for m1, m2 in models: 
        cka_scores = []
        for cp in range(args.check_min, args.check_max+1): 
            cka_mat = torch.load(DATA_DIR / f'cp{cp}_{m1}_and_{m2}.pt').numpy()
            mean_val = (cka_mat[li, lj] + cka_mat[lj, li])/2 
            cka_scores.append(round(mean_val, 4))
        
        ax1.plot(range(args.check_min, args.check_max+1), cka_scores, label=f'{names[m1]} and {names[m2]}', alpha=0.8)


    ax1.set_title(f'CKA between {names[li]} and {names[lj]}')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax1.legend()
    ax1.tick_params(which='both', left=False, bottom=False, top=False, right=False)
    fig1.savefig(EXP_DIR / f'cka_{names[li]}_{names[lj]}.pdf', format='pdf')
    ax1.clear()







