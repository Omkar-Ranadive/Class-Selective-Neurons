"""
Run experiments on sub-checkpoints, i.e model states saved after every kth batch of training in checkpoint i 
"""


from constants import EXP_PATH, IMGNET_PATH, DATA_PATH
import utils 
from ablations import validate
from class_selectivity import get_class_selectivity
import torch 
import torchvision.models as models
from torch import nn
import matplotlib.pyplot as plt 
import argparse 
import os 
import logging 
from datetime import datetime
import numpy as np
import re 


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--check_dir", type=str, required=True) 
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--loader", default='val', type=str)
parser.add_argument("--check_min", required=True, type=int)
parser.add_argument("--check_max", required=True, type=int)
args = parser.parse_args()


"""
Checkpoint structure: 
save_checkpoint({
                ‘epoch’: epoch + 1,
                ‘arch’: args.arch,
                ‘state_dict’: model.state_dict(),
                ‘best_acc1’: best_acc1,
                ‘optimizer’ : optimizer.state_dict(),
            }, is_best)
"""



# Make exp dir if it doesn't exists 
EXP_DIR = EXP_PATH / args.exp_name 
os.makedirs(EXP_DIR, exist_ok=True)

CS_DIR = DATA_PATH / args.check_dir / 'cs_dicts'
os.makedirs(CS_DIR, exist_ok=True)

# Setup logger 
logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info.log'), format='%(asctime)s %(message)s')
logger=logging.getLogger() 

# Path to train dir 
if args.loader == 'train': 
    dir = IMGNET_PATH / 'train'
# Path to validation dir 
else: 
    dir = IMGNET_PATH / 'val'

# Define loss function (criterion)
criterion = nn.CrossEntropyLoss().to('cuda')

loader_cp = utils.load_imagenet_data(dir=dir, batch_size=256, num_workers=args.num_workers)
loader =  utils.load_imagenet_data(dir=dir, batch_size=args.batch_size, num_workers=args.num_workers)

# Max number of channels to ablate based on the layer number (this is based on the model structure)
channels = {4: 256, 5: 512, 6: 1024, 7: 2048}

checkpoints_to_load = [i for i in range(args.check_min, args.check_max)]

# Load pre-trained Resnet 
model = models.resnet50()
model_dict = model.state_dict() 




for f in sorted(os.listdir(DATA_PATH / args.check_dir)):  
    if re.search('e[{}-{}]'.format(args.check_min, args.check_max), f) and 'b' in f: 
        checkpoint = torch.load(DATA_PATH / args.check_dir / f)
        cp = f[:-8]
        # Load checkpoint state dict into the model 
        """
        The key values in checkpoint have different key names, they have an additional "module." in their name 
        Therefore, cleaning the keys before updating state dict of the model 
        """

        for key in checkpoint['state_dict'].keys(): 
            model_key = key.replace("module.", "")
            model_dict[model_key] = checkpoint['state_dict'][key] 

        model.load_state_dict(model_dict)
        model.eval()
        
        # calculate class activations for all the feature maps
        # val_loader = utils.load_imagenet_data(dir=val_dir, batch_size=1, num_workers=8)
        cs_dict_path = CS_DIR / 'cs_dict_{}_{}'.format(args.loader, cp)
        cs_dict_path_full = CS_DIR / 'cs_dict_{}_{}_full'.format(args.loader, cp)

        if not cs_dict_path.is_file(): 
            class_selectivity, class_activations = get_class_selectivity(model=model, val_loader=loader_cp) 
            utils.save_file(class_selectivity, cs_dict_path)
            utils.save_file(class_activations, cs_dict_path_full)
        else: 
            class_selectivity = utils.load_file(cs_dict_path)

        # Run ablations across each of the "sequential-bottleneck layers" (4 to 7)
        for layer in range(4, 8): 
            ablate_dict = {layer: [0, 1, 2, 3, 4, 5]}
            t1_acc = []
            t5_acc = []
            X = list(range(0, channels[layer]+1, 10)) # Stepping the channels to speed it up 
            
            for nc in X: 
                t1, t5 = validate(val_loader=loader, model=model, criterion=criterion, ablate_dict=ablate_dict, num_channels=nc, class_selectivity=class_selectivity)
                logger.info("CP {} Layer number {} Channels ablated {} T1 Acc {:.4f} T5 Acc {:.4f}".format(cp, layer, nc, t1, t5))
                t1_acc.append(t1.item()) 
                t5_acc.append(t5.item()) 

            # For the current layer, plot num channels vs accuracy and save the plot 
            plt.xlabel('Channels ablated')
            plt.ylabel('Accuracy')

            plt.plot(X, t1_acc, label='Top 1 Acc')
            plt.plot(X, t5_acc, label='Top 5 Acc')
            plt.title('Layer {}'.format(layer))
            plt.legend()
            plt.savefig(str(EXP_DIR / '{}_{}_layer_{}.png'.format(datetime.now().strftime('%m_%d_%Y-%H_%M_%S'), cp, layer)))
            plt.clf()

            # Save data for future use 
            np.save(EXP_DIR / 't1_acc_{}_layer_{}'.format(cp, layer), t1_acc) 
            np.save(EXP_DIR / 't5_acc_{}_layer_{}'.format(cp, layer), t5_acc) 
