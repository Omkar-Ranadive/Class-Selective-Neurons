"""
Perform ablations in decreasing order of class selectivity across of the layers of ResNet50 and observe the performance 
"""

from constants import EXP_PATH, IMGNET_PATH 
import utils 
from ablations import validate
from class_selectivity import get_class_selectivity

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

args = parser.parse_args()

# Make exp dir if it doesn't exists 
EXP_DIR = EXP_PATH / args.exp_name 
if not os.path.isdir(EXP_DIR): 
    os.mkdir(EXP_DIR) 

# Setup logger 
logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info.log'), format='%(asctime)s %(message)s')
logger=logging.getLogger() 

# Load pre-trained Resnet 
model = models.resnet50(pretrained=True)
model.eval()

# Path to train dir 
train_dir = IMGNET_PATH / 'train'

# Path to validation dir 
val_dir = IMGNET_PATH / 'val'

# Define loss function (criterion)
criterion = nn.CrossEntropyLoss().to('cuda')
train_loader = utils.load_imagenet_data(dir=val_dir, batch_size=1, num_workers=args.num_workers)  # TODO: Change dir to train_dir later
val_loader = utils.load_imagenet_data(dir=val_dir, batch_size=args.batch_size, num_workers=args.num_workers)


# Max number of channels to ablate based on the layer number (this is based on the model structure)
channels = {4: 256, 5: 512, 6: 1024, 7: 2048}

# calculate class activations for all the feature maps
# val_loader = utils.load_imagenet_data(dir=val_dir, batch_size=1, num_workers=8)
class_selectivity = get_class_selectivity(model=model, val_loader=train_loader) 

# Run ablations across each of the "sequential-bottleneck layers" (4 to 7)
for layer in range(4, 8): 
    ablate_dict = {layer: [0, 1, 2, 3, 4, 5]}
    t1_acc = []
    t5_acc = []
    X = list(range(0, channels[layer]+1, 10)) # Stepping the channels to speed it up 
    

    for nc in X: 
        t1, t5 = validate(val_loader=val_loader, model=model, criterion=criterion, ablate_dict=ablate_dict, num_channels=nc, class_selectivity=class_selectivity)
        logger.info("Layer number {} Channels ablated {} T1 Acc {} T5 Acc {}".format(layer, nc, t1, t5))
        t1_acc.append(t1.item()) 
        t5_acc.append(t5.item()) 

    # For the current layer, plot num channels vs accuracy and save the plot 
    plt.xlabel('Channels ablated')
    plt.ylabel('Accuracy')

    plt.plot(X, t1_acc, label='Top 1 Acc')
    plt.plot(X, t5_acc, label='Top 5 Acc')
    plt.title('Layer {}'.format(layer))
    plt.legend()
    plt.savefig(str(EXP_DIR / '{}_layer_{}.png'.format(datetime.now().strftime('%d_%m_%Y-%H_%M_%S'), layer)))
    plt.clf()

    # Save data for future use 
    np.save(EXP_DIR / 't1_acc_layer_{}'.format(layer), t1_acc) 
    np.save(EXP_DIR / 't5_acc_layer_{}'.format(layer), t5_acc) 









