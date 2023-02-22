from torch_cka import CKA
import torch
from torch import nn
import torchvision.models as models
from constants import *
import utils 
import argparse 
import os 
import logging 
import sys 

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--m1", type=str, required=True)
parser.add_argument("--m2", type=str, required=True)
parser.add_argument("--check_min", type=int, default=0)
parser.add_argument("--check_max", type=int, default=20)
parser.add_argument("--arc", required=True, type=str)

args = parser.parse_args()

layers_m1 = ['layer1', 'layer2', 'layer3', 'layer4', 'fc']
layers_m2 = ['layer1', 'layer2', 'layer3', 'layer4', 'fc']

EXP_DIR = EXP_PATH / args.exp_name 
os.makedirs(EXP_DIR, exist_ok=True)
names = {'layer1': 'Module 4', 'layer2': 'Module 5', 'layer3': 'Module 6', 'layer 4': 'Module 7', 'fc': 'Fully Connected Layer'}

# Setup logger 
logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / f'info{args.m1}_and_{args.m2}.log'), format='%(message)s', filemode='a')
logger=logging.getLogger() 

val_dir = IMGNET_PATH / 'val'
val_loader = utils.load_imagenet_data(dir=val_dir, batch_size=256, num_workers=4)

# Setup the models 
if args.arc == 'resnet50':
        model1 = models.resnet50()
        model2 = models.resnet50()
elif args.arc == 'resnet18': 
        model1 = models.resnet18()
        model2 = models.resnet18()
elif args.arc == 'resnet34': 
        model1 = models.resnet34()
        model2 = models.resnet34()

m1_dict = model1.state_dict() 
m2_dict = model2.state_dict()

# for name, layer in model1.named_modules(): 
#         print(name, layer)

for cp in range(args.check_min, args.check_max+1): 
    logger.info(f"Model1: {args.m1}  Model2: {args.m2} Checkpoint {cp}")  

    M1_DIR = DATA_PATH / args.m1 / f'checkpoint_e{cp}.pth.tar'
    M2_DIR = DATA_PATH / args.m2 / f'checkpoint_e{cp}.pth.tar'

    model1.load_state_dict(utils.load_checkpoint_module(M1_DIR, m1_dict))
    model2.load_state_dict(utils.load_checkpoint_module(M2_DIR, m2_dict))

    cka = CKA(model1, model2,
            model1_name=args.m1,   # good idea to provide names to avoid confusion
            model2_name=args.m2,   
            model1_layers=layers_m1, # List of layers to extract features from
            model2_layers=layers_m2, # extracts all layer features by default
            device='cuda')

    cka.compare(val_loader) # secondary dataloader is optional

    results = cka.export()  # returns a dict that contains model names, layer names
                            # and the CKA matrix


    logger.info(f"CKA: {results['CKA']}  M1 Layers: {results['model1_layers']}  M2 Layers: {results['model2_layers']}")
    logger.info("*"*20)

    # layers_str = "_".join(layers_m1 + layers_m2)
    cka.plot_results(save_path=EXP_DIR / f'cp{cp}_{args.m1}_and_{args.m2}')
    torch.save(results['CKA'],  EXP_DIR / f'cp{cp}_{args.m1}_and_{args.m2}.pt')
    torch.cuda.empty_cache() 

    # utils.save_file(file=results, filename=EXP_DIR / f'{args.m1}_and_{args.m2}_{layers_str}')

