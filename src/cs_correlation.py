from constants import DATA_PATH, EXP_PATH 
import utils 
import torch 
import argparse 
import os 
import logging 
from collections import defaultdict 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from itertools import combinations 

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--check_min", required=True, type=int)
parser.add_argument("--check_max", required=True, type=int)
parser.add_argument("--k", default=10, type=int)
args = parser.parse_args()


EXP_DIR = EXP_PATH / args.exp_name 
os.makedirs(EXP_DIR, exist_ok=True)

# Load imagenet categories 
with open(DATA_PATH / "imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


# Setup logger 
logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info.log'), format='%(message)s', filemode='w')
logger=logging.getLogger() 

num_classes = 1000 
 # Assign each class a different color (1000 color palette)
colors = sns.color_palette('viridis', num_classes) 
color_dict = dict(zip(range(0, num_classes), colors))


selectivities = {}
for layer in [4, 5, 6, 7]: 
    selectivities[layer] = defaultdict(list)

for cp in range(args.check_min, args.check_max+1): 
    logger.info("-"*20)
    logger.info("Checkpoint {}".format(cp))
    class_activations = utils.load_file(DATA_PATH / 'cs_dict_val_cp{}_full'.format(cp))
    # Layer -> Class -> Bottleneck 

    epsilon = 1e-6

    act_vals = defaultdict(list)  # Layer_k -> [[bn1], [bn2]...] where bni includes bottleneck values for each class [1000d mat]
    class_counter = {}
    # Layer_k = outer layer num, layer_v = dict of the form {class_i: {} ... } 
    for layer_k, layer_v in class_activations.items():
        class_counter[layer_k] = defaultdict(int)
        # for class_k, class_v in class_activations[layer_k].items():
        # For a layer, the number of bottleneck layers will be the same 
        # So, just choose any class (in this case class 0) to get the index of bottleneck layers 
        for bottleneck_k, bottleneck_v in class_activations[layer_k][0].items():
            for class_k in sorted(class_activations[layer_k].keys()):
                if class_k > 0:
                    all_activations_for_this_bottleneck = torch.cat((all_activations_for_this_bottleneck, class_activations[layer_k][class_k][bottleneck_k]), dim=0)
                else:
                    all_activations_for_this_bottleneck = class_activations[layer_k][class_k][bottleneck_k]
            
            all_activations_for_this_bottleneck = all_activations_for_this_bottleneck.t()
            print(all_activations_for_this_bottleneck.shape)
            avg_activations_for_bn = torch.mean(all_activations_for_this_bottleneck, dim=0) 
            
            act_vals[layer_k].append(avg_activations_for_bn.numpy())

            u_max_k, u_max_k_indices = torch.topk(avg_activations_for_bn, k=args.k)

            for i, ci in enumerate(u_max_k_indices.numpy()): 
                class_counter[layer_k][ci] += 1 


            # Keep track of class selective index for bottleneck layers 
            u_max, u_max_indices = torch.max(all_activations_for_this_bottleneck, dim=1)
            u_sum = torch.sum(all_activations_for_this_bottleneck, dim=1)
            u_minus_max = (u_sum - u_max) / (all_activations_for_this_bottleneck.shape[1] - 1)

            selectivity = (u_max - u_minus_max) / (u_max + u_minus_max + epsilon)

            avg_selectivity = torch.mean(selectivity)

            # Selectivity at cp 0 will skew the result 
            if cp > 0: 
                selectivities[layer_k][bottleneck_k].append(avg_selectivity.item())
    


    # Save layer wise info to log 
    for layer, counts in class_counter.items(): 
        logger.info("*"*20)
        logger.info("Layer {}".format(layer))  

        for k, v in counts.items(): 
            logger.info("ID {} Name {} Count: {}".format(k, categories[k], v))



    df_counts = pd.DataFrame(class_counter).T
    # bp = sns.barplot(data=df_counts) 
   
    ax = df_counts.plot.bar(rot=0, stacked=True, legend=False, color=color_dict)
    ax.set_title("CP {}: Top categories by layer".format(cp))
    # # for c in ax.containers:
    # #     # Optional: if the segment is small or 0, customize the labels
    # #     labels = df_counts[l]
    # #     # labels = [v.get_height() if v.get_height() > 0 else '' for v in c]
    # #     # remove the labels parameter if it's not needed for customized labels
    # #     ax.bar_label(c, labels=labels, label_type='center')

    plt.savefig(EXP_DIR / 'cp{}_categories_bar.jpg'.format(cp))
    plt.clf()   
    plt.close()


    """
    Form a dataframe from the dict 
    Each column is length 1000 of the form lk_bi for k = layer num and i = bottleneck num 
    """

    df = pd.DataFrame() 
    df_sv = pd.DataFrame() # For selectivity 

    for layer, bns in sorted(act_vals.items()): 
        for i, bn_val in enumerate(bns): 
            df["l{}_b{}".format(layer, i)] = bn_val 

 

    # Plot for all layers 
    hm = sns.heatmap(df.corr())
    hm.set(title="CP {} all layers".format(cp))
    plt.savefig(EXP_DIR / 'cp{}_all_layers.jpg'.format(cp))
    plt.clf()
    plt.close()

    # Plot two layers at a time 
    layers = combinations(act_vals.keys(), 2)
    
    for li, lj in layers: 
        cols_i = ["l{}_b{}".format(li, i) for i in range(len(act_vals[li]))] 
        cols_j = ["l{}_b{}".format(lj, j) for j in range(len(act_vals[lj]))] 

        df_sliced = df[cols_i + cols_j]

        # Plot for the sliced frame 
        hm_sliced = sns.heatmap(df_sliced.corr(), annot=True)
        hm_sliced.set(title="CP {} Layers {} - {}".format(cp, li, lj))
        plt.savefig(EXP_DIR / 'cp{}_layers_{}_{}.jpg'.format(cp, li, lj))
        plt.clf()
    
    plt.close()

df_sv = pd.DataFrame()
for layer, bns in sorted(selectivities.items()): 
    for k, bn_vals in sorted(bns.items()): 
        df_sv["l{}_b{}".format(layer, k)] = bn_vals 


hm = sns.heatmap(df_sv.corr())
hm.set(title="All checkpoints selectivity")
plt.savefig(EXP_DIR / 'All_checkpoints_selectivity_corr.jpg')
plt.clf()
plt.close()


sns.lineplot(data=df_sv)
plt.savefig(EXP_DIR / 'All_checkpoints_selectivity_plot.jpg')
plt.clf()
plt.close()