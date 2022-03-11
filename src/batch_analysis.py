import os 
from constants import DATA_PATH, IMGNET_PATH
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
            

def get_top_classes(class_activations, k=10):
    # Layer -> Class -> Bottleneck 
    epsilon = 1e-6

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
        # logger.info("*"*20)
        # logger.info("Layer {}".format(layer))  
        print("Layer: ", layer)
        for k, v in counts.items(): 
            #logger.info("ID {} Name {} Count: {}".format(k, categories[k], v))
            print(k, categories[k], v)

def gen_batch_dist(bt): 
    """
    INCOMPLETE 
    """
    batch_size = int(bt[0].shape[0])
    total_batches = len(bt)

    # Save class wise distribution : Class Index : [How many times that class occurred in each batch]
    # dist = defaultdict(lambda: [0 for _ in range(total_batches)]) 
    dist = defaultdict(list)
    # for bi, batch in enumerate(bt): 
    #     counts = Counter(batch.numpy()) 
    #     for ci, co in counts.items(): 
    #         dist[ci][bi] += co

    # for bi, batch in enumerate(bt): 
    #     counts = Counter(batch.numpy())
    #     for ci, co in counts.items(): 
    #         dist[ci].append(bi)
    a = []
    for bi, batch in enumerate(bt): 
        a.extend(list(np.unique(batch.numpy())))
        
        print(bi, len(set(a)))
        if bi > 50:
            break 


    # # df = pd.DataFrame.from_dict(dist[ci])
    # # # Change columns to str from int to use with seaborn 
    # # df = df.rename(columns=lambda x: str(x))
    # df = pd.DataFrame(data=dist[1], columns=['1'])
   
    # # sns.displot(data=df, x='999', kind='kde')
    # sns.ecdfplot(data=df, x='1')
    # df2 = pd.DataFrame(data=dist[607], columns=['999'])
    # sns.ecdfplot(data=df2, x='999')

    # plt.savefig(EXP_DIR / 'test.jpg')
    # plt.clf()
    # plt.close()

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--img_dir", type=str, default=IMGNET_PATH)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--loader", default='val', type=str)
    args = parser.parse_args()

    dir = args.img_dir / args.loader
    EXP_DIR = DATA_PATH / args.exp_name 
    loader_cp = utils.load_imagenet_data(dir=dir, batch_size=args.batch_size, num_workers=args.num_workers)

    # Load imagenet categories 
    with open(DATA_PATH / "imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # class_dict = utils.load_file(EXP_DIR / 'cs_dict_cp1_full')
    print("1. Calculate class selectivity for each checkpoint") 
    print("2. Get top classes")
    print("3. Get batch dist")
    choice = int(input()) 
    if choice == 1: 
        cal_cs_dicts()        
    elif choice == 2: 
        # INCOMPLETE 
        get_top_classes(class_dict)
    elif choice == 3: 
        # INCOMPLETE
        bt = torch.load(EXP_DIR / 'bt_e1.pt')
        gen_batch_dist(bt)





