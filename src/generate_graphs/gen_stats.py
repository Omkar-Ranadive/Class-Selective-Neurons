import sys
from pathlib import Path 
src_path = Path(__file__).parent / '../'
sys.path.append(str(src_path))

import matplotlib.pyplot as plt 
import argparse 
import os
from datetime import datetime
import numpy as np
from constants import *
import utils
import seaborn as sns 
import logging 

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--loader", default='val', type=str)
args = parser.parse_args()

EXP_DIR = EXP_PATH / args.exp_name 
os.makedirs(EXP_DIR, exist_ok=True)

channels = {4: 256, 5: 512, 6: 1024, 7: 2048}
check_min = 0 
check_max = 16 
checkpoints_to_load = [i for i in range(check_min, check_max)]

cs_for_every_cp = []

# Setup logger 
logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info.log'), format='%(asctime)s %(message)s', filemode='w')
logger=logging.getLogger() 


for cp in checkpoints_to_load:
    cs_dict_path = DATA_PATH / 'cs_dict_{}_cp{}'.format(args.loader, cp)
    class_selectivity = utils.load_file(cs_dict_path)
    cs_for_every_cp.append(class_selectivity)

for l, c in channels.items():
    print("Processing Layer {}...".format(l))
    LAYER_PATH = EXP_DIR / "layer_{}".format(l)
    os.makedirs(LAYER_PATH, exist_ok=True)
 
    for b in cs_for_every_cp[0][l].keys():
        BOTTLENECK_LAYER_PATH = LAYER_PATH / "bottleneck_layer_{}".format(b)
        os.makedirs(BOTTLENECK_LAYER_PATH, exist_ok=True)
        increased, decreased = 0, 0 
        for i in range(c):
            CHANNEL_PATH = BOTTLENECK_LAYER_PATH / "channel_{}".format(i)
            os.makedirs(CHANNEL_PATH, exist_ok=True)

            cs_for_channel = np.load(CHANNEL_PATH / 'layer{}_bottleneck{}_channel{}_cp{}_to_cp{}.npy'.format(l, b, i, check_min, check_max))
            half = len(cs_for_channel)//2 
            first_half_mean = np.mean(cs_for_channel[ :half])
            second_half_mean = np.mean(cs_for_channel[half: ])

            if second_half_mean > first_half_mean: 
                increased += 1 
            else: 
                decreased += 1 
            

            in_per = increased/(increased+decreased)
            dec_per = 1.0 - in_per 
        
        logger.info("Layer: {}  Bottleneck Layer: {}  CS Increased: {} CS Decreased: {}  % increase {:.2%}  % decrease {:.2%}".format(l, b, increased, decreased, in_per, dec_per))

            


