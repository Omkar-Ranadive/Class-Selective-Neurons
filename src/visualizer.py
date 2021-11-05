import matplotlib.pyplot as plt 
import numpy as np 
from constants import EXP_PATH, DATA_PATH
from datetime import datetime
import argparse 


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
        plt.savefig(str(EXP_DIR_CS / '{}_layer_{}.png'.format(datetime.now().strftime('%m_%d_%Y-%H_%M_%S'), layer)))
        plt.clf()


def cp_vis(): 
    checkpoints = [i for i in range(0, 16)]

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
            plt.savefig(str(EXP_DIR_CS / '{}_cp{}_combined_layer_{}.png'.format(datetime.now().strftime('%m_%d_%Y-%H_%M_%S'), cp, layer)))
            plt.clf()


def pop_vis(): 
    cp = 35

    for layer in range(4, 8):
        t1_top = np.load(EXP_DIR_CS / 't1_acc_cp{}_layer_{}_{}.npy'.format(cp, layer, 'top')) 
        t5_top = np.load(EXP_DIR_CS / 't5_acc_cp{}_layer_{}_{}.npy'.format(cp, layer, 'top')) 

        t1_bot = np.load(EXP_DIR_CS / 't1_acc_cp{}_layer_{}_{}.npy'.format(cp, layer, 'bottom')) 
        t5_bot = np.load(EXP_DIR_CS / 't5_acc_cp{}_layer_{}_{}.npy'.format(cp, layer, 'bottom'))

        X = range(0, cp+1)
        # Plot accuracies for both class selective and random on the same figure 
        plt.xlabel('Checkpoints')
        plt.ylabel('Accuracy')

        plt.plot(X, t1_top, label='T1 (Top)')
        plt.plot(X, t5_top, label='T5 (Top)')
        plt.plot(X, t1_bot, label='T1 (Bottom)')
        plt.plot(X, t5_bot, label='T5 (Bottom)')

        plt.title('Pop compare: Layer {}'.format(layer))
        plt.legend()
        plt.savefig(str(EXP_DIR_CS / '{}_cp{}_combined_layer_{}.png'.format(datetime.now().strftime('%m_%d_%Y-%H_%M_%S'), cp, layer)))
        plt.clf()



if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name_ran", type=str, required=True)
    parser.add_argument("--exp_name_cs", type=str, required=True)
    parser.add_argument("--vis", default="nor", type=str)
    args = parser.parse_args()

    EXP_DIR_RAN = EXP_PATH / args.exp_name_ran
    EXP_DIR_CS = EXP_PATH / args.exp_name_cs 

    # Max number of channels to ablate based on the layer number (this is based on the model structure)
    channels = {4: 256, 5: 512, 6: 1024, 7: 2048}

    
    if args.vis == 'nor': 
        normal_vis() 
    elif args.vis == 'cp': 
        cp_vis() 
    elif args.vis == 'pop':
        pop_vis()