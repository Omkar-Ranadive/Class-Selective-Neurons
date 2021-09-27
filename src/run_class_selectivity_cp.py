from numpy.lib import stride_tricks
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


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--loader", default='val', type=str)
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
if not os.path.isdir(EXP_DIR): 
    os.mkdir(EXP_DIR) 

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

checkpoints_to_load = [i for i in range(1, 6)]

# Load pre-trained Resnet 
model = models.resnet50()
model_dict = model.state_dict() 


for cp in checkpoints_to_load: 
    checkpoint = torch.load(DATA_PATH / 'model_checkpoints' / 'CHECKPOINTS' / 'EXPE1' / 'checkpoint_epoch{}.pth.tar'.format(cp))

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
    cs_dict_path = DATA_PATH / 'cs_dict_{}_cp{}'.format(args.loader, cp)
    if not cs_dict_path.is_file(): 
        class_selectivity = get_class_selectivity(model=model, val_loader=loader_cp) 
        utils.save_file(class_selectivity, DATA_PATH / 'cs_dict_{}_cp{}'.format(args.loader, cp))
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
            logger.info("CP {} Layer number {} Channels ablated {} T1 Acc {} T5 Acc {}".format(cp, layer, nc, t1, t5))
            t1_acc.append(t1.item()) 
            t5_acc.append(t5.item()) 

        # For the current layer, plot num channels vs accuracy and save the plot 
        plt.xlabel('Channels ablated')
        plt.ylabel('Accuracy')

        plt.plot(X, t1_acc, label='Top 1 Acc')
        plt.plot(X, t5_acc, label='Top 5 Acc')
        plt.title('Layer {}'.format(layer))
        plt.legend()
        plt.savefig(str(EXP_DIR / '{}_cp{}_layer_{}.png'.format(datetime.now().strftime('%d_%m_%Y-%H_%M_%S'), cp, layer)))
        plt.clf()

        # Save data for future use 
        np.save(EXP_DIR / 't1_acc_cp{}_layer_{}'.format(cp, layer), t1_acc) 
        np.save(EXP_DIR / 't5_acc_cp{}_layer_{}'.format(cp, layer), t5_acc) 
