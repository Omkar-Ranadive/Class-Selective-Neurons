import os 
from constants import DATA_PATH, IMGNET_PATH, EXP_PATH
import argparse 
import torchvision.models as models
import re 
import utils 
from class_selectivity import get_class_selectivity
from collections import defaultdict, Counter
import torch 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import numpy as np 

def cal_cs_dicts(): 
    model = models.resnet50()
    model_dict = model.state_dict() 

    for f in os.listdir(EXP_DIR): 
        if 'checkpoint' in f: 
            epoch_num = re.search(r'\d+', f).group()
            cs_dict_path = EXP_DIR / 'cs_dict_{}_cp{}'.format(args.loader, epoch_num)

            model_dict = utils.load_checkpoint_module(CHECK_DIR=EXP_DIR / f, model_dict=model_dict)
            model.load_state_dict(model_dict)
            model.eval()

            if not cs_dict_path.is_file(): 
                class_selectivity, class_activations = get_class_selectivity(model=model, val_loader=loader_cp) 
                utils.save_file(class_selectivity, EXP_DIR / 'cs_dict_{}_cp{}'.format(args.loader, epoch_num))
                utils.save_file(class_activations, EXP_DIR / 'cs_dict_{}_cp{}_full'.format(args.loader, epoch_num))
            

def get_top_classes(k=200):
    # Layer -> Class -> Bottleneck 
    epsilon = 1e-6
    sns.set_theme()

    for f in sorted(os.listdir(EXP_DIR)):
        if 'full' in f: 
            print(f"Calculating for {f}")
            class_activations =  utils.load_file(EXP_DIR / f)
            selectivities = {}
            for layer in [4, 5, 6, 7]: 
                selectivities[layer] = defaultdict(list)

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
                    avg_activations_for_bn = torch.mean(all_activations_for_this_bottleneck, dim=0) 
                    
                    act_vals[layer_k].append(avg_activations_for_bn.numpy())

                    u_max_k, u_max_k_indices = torch.topk(avg_activations_for_bn, k=k)

                    for i, ci in enumerate(u_max_k_indices.numpy()): 
                        class_counter[layer_k][ci] += 1 


                    # Keep track of class selective index for bottleneck layers 
                    u_max, u_max_indices = torch.max(all_activations_for_this_bottleneck, dim=1)
                    u_sum = torch.sum(all_activations_for_this_bottleneck, dim=1)
                    u_minus_max = (u_sum - u_max) / (all_activations_for_this_bottleneck.shape[1] - 1)

                    selectivity = (u_max - u_minus_max) / (u_max + u_minus_max + epsilon)

                    avg_selectivity = torch.mean(selectivity)

                    # # Selectivity at cp 0 will skew the result 
                    # if cp > 0: 
                    #     selectivities[layer_k][bottleneck_k].append(avg_selectivity.item())

            for layer, counts in class_counter.items(): 
                    plt.bar(counts.keys(), counts.values(), edgecolor='#43A6C6')
                    plt.savefig(SAVE_DIR / f'top_{k}_layer_{layer}_{f}.jpg')
                    plt.clf()
                    plt.close()


def gen_batch_dist(thres=0.3): 
    sns.set_theme()
 
    for f in sorted(os.listdir(EXP_DIR)): 
        if 'bt' in f: 
            print(f"Calculating for batch {f}")
            bt = torch.load(EXP_DIR / f) 
            total_batches = len(bt)
            num_batches = int(thres * total_batches)  # Select the first thres num of batches and see how classes are distributed within this 

            dist = defaultdict(list)
            classes = []
            for bi, batch in enumerate(bt): 
                classes.extend(list(batch.numpy()))
                
                if bi > num_batches:
                    break 

            cdict = Counter(classes)
            plt.bar(cdict.keys(), cdict.values(), edgecolor='#43A6C6')

            plt.savefig(SAVE_DIR / f'batch_dist_{f[:-3]}_t{thres}.jpg')
            plt.clf()

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--img_dir", type=str, default=IMGNET_PATH)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--loader", default='val', type=str)
    parser.add_argument("--save_dir", default=None, type=str)
    parser.add_argument("--thres", default=0.3, type=float)
    parser.add_argument("--k", default=200, type=int)
    args = parser.parse_args()

    dir = args.img_dir / args.loader
    EXP_DIR = DATA_PATH / args.exp_name 
    loader_cp = utils.load_imagenet_data(dir=dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # Load imagenet categories 
    with open(DATA_PATH / "imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    if not args.save_dir:
        SAVE_DIR = EXP_DIR
    else: 
        SAVE_DIR = EXP_PATH / args.save_dir 
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Calculate class selectivity for each checkpoint") 
    print("2. Get top classes")
    print("3. Get batch dist")
    choice = int(input()) 
    if choice == 1: 
        cal_cs_dicts()        
    elif choice == 2: 
        get_top_classes(k=args.k)
    elif choice == 3: 
        gen_batch_dist(thres=args.thres)





