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
from matplotlib.ticker import MaxNLocator, LinearLocator


def compare_cp_acc(dirs, keys):

    dir_dict = {'rn18_a20e0': 'Regularized from epoch 0 onward', 'rn18_a20e5': 'Regularized from epoch 5 onward', 
                'rn34_a20e0': 'Regularized from epoch 0 onward', 'rn34_a20e5': 'Regularized from epoch 5 onward'}
    keys_dict = {'train_acc1': 'Train Accuracy', 'train_acc5': 'train_acc_5', 'val_acc1': 'Val Accuracy', 'val_acc5': 'val_acc5'}

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors[0], colors[1] = colors[1], colors[0] 

    for key in keys: 
        fig, ax = plt.subplots() 
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

        for i, dir in enumerate(dirs): 
            DATA_DIR = DATA_PATH / dir 
            accs = []
            files = natsort.natsorted(os.listdir(DATA_DIR))

            for cp in files: 
                if '.tar' in cp: 
                    if re.search('e\d+', cp): 
                        cp_num = re.search('e\d+', cp).group()[1:]
                        if args.check_min is not None and args.check_max is not None: 
                            if args.check_min <= int(cp_num) <= args.check_max: 
                                cur_cp = torch.load(DATA_DIR / cp)
                                acc = cur_cp[key] if isinstance(cur_cp[key], int) else cur_cp[key].item()
                                accs.append(acc)
                        else: 
                            cur_cp = torch.load(DATA_DIR / cp)
                            acc = cur_cp[key] if isinstance(cur_cp[key], int) else cur_cp[key].item()
                            accs.append(acc)
                
            ax.plot(range(0, len(accs)), accs, label=f'{dir_dict[dir]}', alpha=0.8, c=colors[i])
        
        ax.set_title(f"{keys_dict[key]}")
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        if args.check_min is None and args.check_max is None: 
            fig.savefig(SAVE_DIR / f'cp_{key}_com.pdf', format='pdf')
        else: 
            fig.savefig(SAVE_DIR / f'cp_{args.check_min}_to_{args.check_max}_{key}_com.pdf', format='pdf')

        plt.close(fig)
 


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--check_min", type=int, default=None)
    parser.add_argument("--check_max", type=int, default=None)
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--k", default=None, type=str)
    parser.add_argument('-l','--list', nargs='+')
    
    args = parser.parse_args()

    sns.set_theme(style='whitegrid')
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.titlesize'] = 'large'
    plt.rcParams['axes.labelsize'] = 'medium'


    SAVE_DIR = EXP_PATH / args.save_dir 
    os.makedirs(SAVE_DIR, exist_ok=True)


    # Max number of channels to ablate based on the layer number (this is based on the model structure)
    channels = {4: 256, 5: 512, 6: 1024, 7: 2048}
    keys = ['train_acc1', 'train_acc5', 'val_acc1', 'val_acc5']
    compare_cp_acc(dirs=args.list, keys=keys)
