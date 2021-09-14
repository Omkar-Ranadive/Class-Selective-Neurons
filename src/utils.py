from torchvision import transforms
import torchvision.datasets as datasets
import torch 

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
