from constants import DATA_PATH, EXP_PATH 
import utils 
import torch 
import argparse 
import os 
import logging 


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

class_activations = utils.load_file(DATA_PATH / 'cs_dict_val_cp5_full')
# Layer -> Class -> Bottleneck 

epsilon = 1e-6

# Layer_k = outer layer num, layer_v = dict of the form {class_i: {} ... } 
for layer_k, layer_v in class_activations.items():
    logger.info("*"*20)
    logger.info("Layer {}".format(layer_k))
    # for class_k, class_v in class_activations[layer_k].items():
    # For a layer, the number of bottleneck layers will be the same 
    # So, just choose any class (in this case class 0) to get the index of bottleneck layers 
    for bottleneck_k, bottleneck_v in class_activations[layer_k][0].items():
        logger.info("-"*20)
        logger.info("Bottleneck Layer: {}".format(bottleneck_k))
        for class_k in sorted(class_activations[layer_k].keys()):
            if class_k > 0:
                all_activations_for_this_bottleneck = torch.cat((all_activations_for_this_bottleneck, class_activations[layer_k][class_k][bottleneck_k]), dim=0)
            else:
                all_activations_for_this_bottleneck = class_activations[layer_k][class_k][bottleneck_k]
        
        all_activations_for_this_bottleneck = all_activations_for_this_bottleneck.t()
        avg_activations_for_bn = torch.mean(all_activations_for_this_bottleneck, dim=0) 
        
        u_max_k, u_max_k_indices = torch.topk(avg_activations_for_bn, k=args.k)
                
        logger.info("Top {} categories: ".format(args.k)) 
        for i in range(args.k): 
            logger.info("ID: {} Name: {} Val: {:.4f}".format(u_max_k_indices[i], categories[u_max_k_indices[i]],
                                                         u_max_k[i]))
        


        # print("For Layer: {}, shape: {}".format(layer_k, all_activations_for_this_bottleneck.shape))
        # u_max, u_max_indices = torch.max(all_activations_for_this_bottleneck, dim=1)
        # print("UMAX: ", u_max.shape)
        # u_sum = torch.sum(all_activations_for_this_bottleneck, dim=1)
        # u_minus_max = (u_sum - u_max) / (all_activations_for_this_bottleneck.shape[1] - 1)

        # selectivity = (u_max - u_minus_max) / (u_max + u_minus_max + epsilon)
        
        # class_selectivity[layer_k].update({bottleneck_k: selectivity})