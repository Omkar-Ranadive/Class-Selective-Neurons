import sys
from pathlib import Path 
src_path = Path(__file__).parent / '../'
sys.path.append(str(src_path))


import matplotlib.pyplot as plt 
import numpy as np 
from constants import EXP_PATH, DATA_PATH
from datetime import datetime
import argparse 
import os 
import re 
import torch
import natsort 
import seaborn as sns 


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--check_min", required=True, type=int)
parser.add_argument("--check_max", required=True, type=int)
parser.add_argument("--key", type=str, required=True)
parser.add_argument("--format", type=str, default='pdf')
parser.add_argument("--arc", type=str, required=True)
parser.add_argument("--dpi", default=300, type=int)
args = parser.parse_args()


EXP_DIR = EXP_PATH / args.exp_name 
os.makedirs(EXP_DIR, exist_ok=True)


sns.set_theme(style='whitegrid')
plt.rcParams['savefig.dpi'] = args.dpi 
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors[0], colors[2] = colors[2], colors[0]  


fig1, ax1 = plt.subplots()
ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=6, integer=True))

if args.arc == 'resnet50':
    dirs_og = ['rn50_1', 'rn50_2', 'rn50_3', 'rn50_4', 'rn50_5']
    dirs_e0 = ['rn50_a20e0_1', 'rn50_a20e0_2', 'rn50_a20e0_3', 'rn50_a20e0_4', 'rn50_a20e0_5']
    dirs_e5 = ['rn50_a20e5_1', 'rn50_a20e5_2', 'rn50_a20e5_3', 'rn50_a20e5_4', 'rn50_a20e5_5']
    all_dirs = [dirs_og, dirs_e0, dirs_e5]
elif args.arc == 'vgg16': 
    dirs_og = ['vgg16_1', 'vgg16_2', 'vgg16_3', 'vgg16_4', 'vgg16_5']
    all_dirs = [dirs_og]

names = ['Original unregularized model', 'Regularized from epoch 0 onward', 'Regularized from epoch 5 onward']
key_names = {'train_acc1': 'Train Accuracy', 'val_acc1': 'Val Accuracy', 'val_acc5': 'Val Accuracy', 'train_acc5': 'Train Accuracy'}
markers = ['o', '^', 'X']


for index, dirs in enumerate(all_dirs): 
    all_accs = []
    print("Dir len", len(dirs))
    for dir in dirs: 
        DATA_DIR = DATA_PATH / dir 
        accs = []
        files = natsort.natsorted(os.listdir(DATA_DIR))
        for cp in files: 
            if '.tar' in cp: 
                if re.search('e\d+', cp): 
                    cp_num = re.search('e\d+', cp).group()[1:]
                    if args.check_min <= int(cp_num) <= args.check_max: 
                        cur_cp = torch.load(DATA_DIR / cp)
                        acc = cur_cp[args.key] if (isinstance(cur_cp[args.key], int) or isinstance(cur_cp[args.key], float)) else cur_cp[args.key].item()
                        accs.append(acc)
        all_accs.append(accs)

    all_accs = np.array(all_accs)
    avg_acc = np.mean(all_accs, axis=0)
    se = np.std(all_accs, axis=0) / np.sqrt(len(dirs))
    confidence_intervals = 2 * se        
    ax1.plot(range(args.check_min, args.check_max+1), avg_acc, label=f'{names[index]}', c=colors[index], marker=markers[index], alpha=0.8)
    ax1.fill_between(range(args.check_min, args.check_max+1), avg_acc - confidence_intervals, avg_acc + confidence_intervals, 
                     alpha=0.3, color=colors[index])


ax1.set_title(f"{key_names[args.key]}")
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

fig1.savefig(EXP_DIR / f'cp_{args.check_min}_to_{args.check_max}_{args.key}_com.{args.format}', format=args.format, bbox_inches='tight')

