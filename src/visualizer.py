import matplotlib.pyplot as plt 
import numpy as np 
from constants import EXP_PATH, DATA_PATH
from datetime import datetime
import argparse 
import os 
import re 

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


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name_ran", type=str, required=True)
    parser.add_argument("--exp_name_cs", type=str, required=True)
    parser.add_argument("--vis", default="nor", type=str)
    parser.add_argument("--check_min", required=True, type=int)
    parser.add_argument("--check_max", required=True, type=int)
    parser.add_argument("--save_dir", default=None, type=str)
    parser.add_argument("--split_per", default=None, type=float)
    args = parser.parse_args()

    EXP_DIR_RAN = EXP_PATH / args.exp_name_ran
    EXP_DIR_CS = EXP_PATH / args.exp_name_cs 
    if args.save_dir is not None: 
        SAVE_DIR = EXP_PATH / args.save_dir 
    else: 
        SAVE_DIR = EXP_DIR_CS

    os.makedirs(SAVE_DIR, exist_ok=True)

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
