from pathlib import Path

PARENT_PATH = Path(__file__).parent
# DATA_PATH = PARENT_PATH / '../data'
DATA_PATH = Path('/mnt/data/omkar/CSN/data')
IMGNET_PATH = Path('/data/ImageNet/')
EXP_PATH = PARENT_PATH / '../experiments' 


IMGNET_CLASSES = 1000 
EPSILON = 1e-6
