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


sns.set_theme()


def normal_vis(): 
    for layer in range(4, 8): 

        X = list(range(0, channels[layer]+1, 10)) # Stepping the channels to speed it up 

        t1_ran = np.load(EXP_DIR_RAN / 't1_acc_layer_{}.npy'.format(layer))
        t5_ran = np.load(EXP_DIR_RAN / 't5_acc_layer_{}.npy'.format(layer))

        t1_cs = np.load(EXP_DIR_CS / 't1_acc_layer_{}.npy'.format(layer))
        t5_cs = np.load(EXP_DIR_CS / 't5_acc_layer_{}.npy'.format(layer))

        # Plot accuracies for both class selective and random on the same figure 
        plt.xlabel('Channels ablated')
        plt.ylabel('Accuracy')

        plt.plot(X, t1_ran, label='Top 1 Acc Ran')
        plt.plot(X, t5_ran, label='Top 5 Acc Ran')
        plt.plot(X, t1_cs, label='Top 1 Acc CS')
        plt.plot(X, t5_cs, label='Top 5 Acc CS')

        plt.title('Layer {}'.format(layer))
        plt.legend()
        plt.savefig(str(SAVE_DIR / '{}_layer_{}.png'.format(datetime.now().strftime('%m_%d_%Y-%H_%M_%S'), layer)))
        plt.clf()


def cp_vis(): 
    checkpoints = [i for i in range(args.check_min, args.check_max)]

    for cp in checkpoints: 
        for layer in range(4, 8): 

            X = list(range(0, channels[layer]+1, 10)) # Stepping the channels to speed it up 

            t1_ran = np.load(EXP_DIR_RAN / 't1_acc_cp{}_layer_{}.npy'.format(cp, layer))
            t5_ran = np.load(EXP_DIR_RAN / 't5_acc_cp{}_layer_{}.npy'.format(cp, layer))

            t1_cs = np.load(EXP_DIR_CS / 't1_acc_cp{}_layer_{}.npy'.format(cp, layer))
            t5_cs = np.load(EXP_DIR_CS / 't5_acc_cp{}_layer_{}.npy'.format(cp, layer))

            # Plot accuracies for both class selective and random on the same figure 
            plt.xlabel('Channels ablated')
            plt.ylabel('Accuracy')

            plt.plot(X, t1_ran, label='Top 1 Acc Ran')
            plt.plot(X, t5_ran, label='Top 5 Acc Ran')
            plt.plot(X, t1_cs, label='Top 1 Acc CS')
            plt.plot(X, t5_cs, label='Top 5 Acc CS')

            plt.title('CP {} Layer {}'.format(cp, layer))
            plt.legend()
            plt.savefig(str(SAVE_DIR / '{}_cp{}_combined_layer_{}.png'.format(datetime.now().strftime('%m_%d_%Y-%H_%M_%S'), cp, layer)))
            plt.clf()


def pop_vis(): 

    for layer in range(4, 8):
        t1_top = np.load(EXP_DIR_CS / 't1_acc_cp{}_cp{}_layer_{}_{}_split_{}.npy'.format(args.check_min,
         args.check_max, layer, 'top', args.split_per)) 
        t5_top = np.load(EXP_DIR_CS / 't5_acc_cp{}_cp{}_layer_{}_{}_split_{}.npy'.format(args.check_min,
         args.check_max, layer, 'top', args.split_per)) 

        t1_bot = np.load(EXP_DIR_CS / 't1_acc_cp{}_cp{}_layer_{}_{}_split_{}.npy'.format(args.check_min,
         args.check_max, layer, 'bottom', args.split_per)) 
        t5_bot = np.load(EXP_DIR_CS / 't5_acc_cp{}_cp{}_layer_{}_{}_split_{}.npy'.format(args.check_min,
         args.check_max, layer, 'bottom', args.split_per)) 

        X = range(0, args.check_max)

        # Plot accuracies for both class selective and random on the same figure 
        plt.xlabel('Checkpoints')
        plt.ylabel('Accuracy')

        plt.plot(X, t1_top, label='T1 (Top ablated)')
        plt.plot(X, t5_top, label='T5 (Top ablated)')
        plt.plot(X, t1_bot, label='T1 (Bottom ablated)')
        plt.plot(X, t5_bot, label='T5 (Bottom ablated)')

        plt.title('Pop compare: Layer {} Split {}%'.format(layer, args.split_per*100))
        plt.legend()
        plt.savefig(str(SAVE_DIR / '{}_cp{}_cp{}_combined_pop_layer_{}.png'.format(datetime.now().strftime('%m_%d_%Y-%H_%M_%S'),  
        args.check_min, args.check_max, layer)))
        
        plt.clf()

def cpb_vis(): 
    for f in sorted(os.listdir(EXP_DIR_CS)):  
        matches = ['.npy', 'b', 't1']
        if all(x in f for x in matches) and re.search('e[{}-{}]'.format(args.check_min, args.check_max), f):
            t1_ran = np.load(EXP_DIR_RAN / f)
            t5_ran = np.load(EXP_DIR_RAN / f.replace('t1', 't5'))

            t1_cs = np.load(EXP_DIR_CS / f)
            t5_cs = np.load(EXP_DIR_CS / f.replace('t1', 't5'))

            X = list(range(0, channels[int(f[-5])]+1, 10)) # Stepping the channels to speed it up, note f[-5] gives the layer num

            # Plot accuracies for both class selective and random on the same figure 
            plt.xlabel('Channels ablated')
            plt.ylabel('Accuracy')

            plt.plot(X, t1_ran, label='Top 1 Acc Ran')
            plt.plot(X, t5_ran, label='Top 5 Acc Ran')
            plt.plot(X, t1_cs, label='Top 1 Acc CS')
            plt.plot(X, t5_cs, label='Top 5 Acc CS')

            plt.title('{}'.format(f[:-4]))
            plt.legend()
            plt.savefig(str(SAVE_DIR / '{}_combined_{}.png'.format(datetime.now().strftime('%m_%d_%Y-%H_%M_%S'), f[:-4])))
            plt.clf()

def combine_plots_from_dirs(dirs, rows=2, cols=2): 
    fig = plt.figure(figsize=(10, 7))
    imgs = []
    for dir in dirs: 
        EXP_DIR = EXP_PATH / dir 
        temp = []
        for f in sorted(os.listdir(EXP_DIR)): 
            img = plt.imread(EXP_DIR / f) 
            temp.append((img, f[:-4])) 
        
        imgs.append(temp)

    for i in range(len(imgs[0])): 
        for j, dir_name in enumerate(dirs): 
            fig.add_subplot(rows, cols, j+1)
            plt.imshow(imgs[j][i][0])
            plt.axis('off')
            plt.title(dir_name, fontdict = {'fontsize' : 6})

        fig.savefig(SAVE_DIR / '{}.png'.format(imgs[j][i][1]), dpi=300, bbox_inches = 'tight', pad_inches = 0.1)
        fig.clear()


def compare_cp_acc(dirs, key):

    for dir in dirs: 
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
                            acc = cur_cp[key] if (isinstance(cur_cp[key], float) or isinstance(cur_cp[key], int)) else cur_cp[key].item()
                            accs.append(acc)
                    else: 
                        cur_cp = torch.load(DATA_DIR / cp)
                        if (isinstance(cur_cp[key], float) or isinstance(cur_cp[key], int)):
                            acc = cur_cp[key] 
                        else:
                            cur_cp[key].item()
                        accs.append(acc)
            
        plt.plot(range(0, len(accs)), accs, label='{}'.format(dir), alpha=0.8)
    
    plt.title(f"{key}")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    if args.check_min is None and args.check_max is None: 
        plt.savefig(SAVE_DIR / f'cp_{key}_com.pdf', format='pdf')
    else: 
        plt.savefig(SAVE_DIR / f'cp_{args.check_min}_to_{args.check_max}_{key}_com.pdf', format='pdf')


 


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name_ran", default="dummy_ran", type=str)
    parser.add_argument("--exp_name_cs", default="dummy_cs", type=str)
    parser.add_argument("--vis", default="compcp", type=str)
    parser.add_argument("--check_min", type=int, default=None)
    parser.add_argument("--check_max", type=int, default=None)
    parser.add_argument("--save_dir", default=None, type=str)
    parser.add_argument("--split_per", default=None, type=float)
    parser.add_argument("--k", default=None, type=str)
    parser.add_argument('-l','--list', nargs='+')
    
    args = parser.parse_args()

    sns.set_theme(style='whitegrid')
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.titlesize'] = 'large'
    plt.rcParams['axes.labelsize'] = 'medium'

    EXP_DIR_RAN = EXP_PATH / args.exp_name_ran
    EXP_DIR_CS = EXP_PATH / args.exp_name_cs 
    if args.save_dir is not None: 
        SAVE_DIR = EXP_PATH / args.save_dir 
        os.makedirs(SAVE_DIR, exist_ok=True)
    else: 
        SAVE_DIR = EXP_DIR_CS


    # Max number of channels to ablate based on the layer number (this is based on the model structure)
    channels = {4: 256, 5: 512, 6: 1024, 7: 2048}

    if args.vis == 'nor': 
        normal_vis() 
    elif args.vis == 'cp': 
        cp_vis() 
    elif args.vis == 'pop':
        pop_vis()
    elif args.vis == 'cpb': 
        cpb_vis() 
    elif args.vis == 'combd':
        combine_plots_from_dirs(dirs=args.list)
    elif args.vis == 'compcp':
        compare_cp_acc(dirs=args.list, key=args.k)
