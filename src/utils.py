from torchvision import transforms
import torchvision.datasets as datasets
import torch 
from constants import DATA_PATH
import pickle 


class AverageMeter(object):
    """
    Credits: https://github.com/pytorch/examples/blob/master/imagenet/main.py

    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    Credits: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'





def process_img(img):
    """
    Process the data the way the Resnet50 model requires it 

    Args:
        img ([type]): [description]
    """

    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return preprocess(img)


def load_imagenet_data(dir, batch_size, num_workers): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(dir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True)
    

    return loader 


def save_file(file, filename, path=DATA_PATH):
    """
    Save file in pickle format
    Args:
        file (any object): Can be any Python object. We would normally use this to save the
        processed Pytorch dataset
        filename (str): Name of the file
        path (Path obj): Path to save file to
    """

    with open(path / filename, 'wb') as f:
        pickle.dump(file, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(filename, path=DATA_PATH):
    """
    Load a pickle file
    Args:
        filename (str): Name of the file
        path (Path obj): Path to load file from
    Returns (Python obj): Returns the loaded pickle file
    """
    with open(path / filename, 'rb') as f:
        file = pickle.load(f)

    return file