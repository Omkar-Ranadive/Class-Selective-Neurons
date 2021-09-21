import matplotlib.pyplot as plt 
import numpy as np 
from constants import EXP_PATH, DATA_PATH
from datetime import datetime
import argparse 


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name_ran", type=str, required=True)
parser.add_argument("--exp_name_cs", type=str, required=True)
args = parser.parse_args()

EXP_DIR_RAN = EXP_PATH / args.exp_name_ran
EXP_DIR_CS = EXP_PATH / args.exp_name_cs 

# Max number of channels to ablate based on the layer number (this is based on the model structure)
channels = {4: 256, 5: 512, 6: 1024, 7: 2048}

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
    plt.savefig(str(EXP_DIR_CS / '{}_layer_{}.png'.format(datetime.now().strftime('%d_%m_%Y-%H_%M_%S'), layer)))
    plt.clf()

    
