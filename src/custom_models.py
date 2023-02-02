import torch
from torch import nn
import torchvision.models as models


class NewModel(nn.Module):
    def __init__(self, arch, pretrained=False, *args):
        super().__init__(*args)
        self.features = {}

        self.model = models.__dict__[arch](pretrained=pretrained)

        if arch == 'resnet50': 
            self.return_nodes = {
                            'layer1.0.relu_2': 4, 'layer1.1.relu_2': 4,'layer1.2.relu_2': 4,
                            'layer2.0.relu_2': 5,'layer2.1.relu_2': 5, 'layer2.2.relu_2': 5, 'layer2.3.relu_2': 5,
                            'layer3.0.relu_2': 6, 'layer3.1.relu_2':6, 'layer3.2.relu_2': 6, 'layer3.3.relu_2': 6, 'layer3.4.relu_2': 6, 'layer3.5.relu_2': 6,
                            'layer4.0.relu_2': 7, 'layer4.1.relu_2': 7, 'layer4.2.relu_2': 7
                        }
        elif arch == 'resnet18': 
            self.return_nodes = {
                            'layer1.0.relu_1': 4, 'layer1.1.relu_1': 4,
                            'layer2.0.relu_1': 5, 'layer2.1.relu_1': 5,
                            'layer3.0.relu_1': 6, 'layer3.1.relu_1': 6,
                            'layer4.0.relu_1': 7, 'layer4.1.relu_1': 7
                        }
        elif arch == 'resnet34': 
            self.return_nodes = {
                            'layer1.0.relu_1': 4, 'layer1.1.relu_1': 4, 'layer1.2.relu_1': 4,
                            'layer2.0.relu_1': 5, 'layer2.1.relu_1': 5, 'layer2.2.relu_1': 5, 'layer2.3.relu_1': 5,
                            'layer3.0.relu_1': 6, 'layer3.1.relu_1': 6, 'layer3.2.relu_1': 6, 'layer3.3.relu_1': 6, 'layer3.4.relu_1': 6, 'layer3.5.relu_1': 6,
                            'layer4.0.relu_1': 7, 'layer4.1.relu_1': 7, 'layer4.2.relu_1': 7
                        }               

        # Generate hooks for the points of interest (i.e, output point at which we calculate selectivity)
        layers = [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        for k in self.return_nodes.keys():
            splits = k.split(".")
            l_num = int(splits[0][-1])
            bn_num = int(splits[1])
            layers[l_num-1][bn_num].relu.register_forward_hook(self.forward_hook(k))


    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook


    def forward(self, x):
        out = self.model(x)
        return out, self.features

