import torch
import torchvision
import torchvision.transforms as transforms
from constants import DATA_PATH
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse 
import shutil
import os 
from class_selectivity_v3 import get_class_selectivity, forward_grad, get_class_selectivity_cus_model
import utils 
import re
import logging 
from custom_models import VGGModel


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, str(EXP_DIR / filename))
    if is_best:
        shutil.copyfile(str(EXP_DIR / filename), str(EXP_DIR / 'model_best.pth.tar'))


def validate(model):
    correct = 0
    total = 0
    model.eval()

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs, _ = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100*(correct / total) 

    return acc


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--alpha", required=True, type=str, help=
                    """ It should be a expression of the form alpha_val_i * num_epochs alpha_val_2 * num_epochs .....
                    Example: To run with alpha = 0 for first 5 epochs and then alpha = -20 for next 15 epochs, expression would be: " 0*5 -20*15"
                    (Make sure to quote around the string to avoid parsing errors. 
                    If starting with a negative value, for example - " -20*20", make sure to add a space after quote to avoid negative number issues in Linux) 
                    """)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--mom", type=float, default=0.9)

args = parser.parse_args()

# Find all occurrences of a number followed by "*" and another number
matches = re.findall(r'(-?\d+)\s*\*\s*(-?\d+)', args.alpha)
args.alphas = []
# Print the matched numbers
for match in matches:
    alpha = float(match[0])
    multiplier = int(match[1]) 
    args.alphas.extend([alpha]*multiplier)

EXP_DIR = DATA_PATH / args.exp_name
os.makedirs(EXP_DIR, exist_ok=True)
args.arc = 'vgg16' 
loader = 'val'

logging.basicConfig(level=logging.INFO, filename=str(EXP_DIR / 'info.log'), format='%(message)s', filemode='a')
logger = logging.getLogger()

logger.info(f'Batch size: {args.batch_size}')
logger.info(f'Training epochs: {args.epochs}')
# logger.info(f'Learning Rate: {args.lr}')
logger.info(f'Model architecture: {args.arc}')
logger.info(f'Alpha: {args.alpha}')
logger.info(f'Alphas: {args.alphas}')
logger.info(f"Learning Rate: {args.lr}")
logger.info(f"Momentum: {args.mom}")


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = args.batch_size

trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

args.return_nodes = {
                'features.1': 1, 'features.3': 1, 
                'features.6': 2, 'features.8': 2,
                'features.11': 3, 'features.13': 3, 'features.15': 3,
                'features.18': 4, 'features.20': 4, 'features.22': 4,
                'features.25': 5, 'features.27': 5, 'features.29': 5,
                'classifier.1': 6, 'classifier.4': 6
            }



epsilon = 1e-6
model = VGGModel(arch=args.arc)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)

best_acc1 = 0
acc1 = validate(model)

class_selectivity, class_activations = get_class_selectivity_cus_model(checkpoint=model, val_loader=testloader) 
utils.save_file(class_selectivity, EXP_DIR / 'cs_dict_{}_cp{}'.format(loader, 0))
utils.save_file(class_activations, EXP_DIR / 'cs_dict_{}_cp{}_full'.format(loader, 0))

save_checkpoint({
            'epoch': 0,
            'arc': args.arc,
            #'state_dict': model.state_dict(),
            'best_acc1': acc1,
            'val_acc1': acc1, 
            'train_acc1': 0,
            #'optimizer' : optimizer.state_dict(),
        }, False, filename='checkpoint_e{}.pth.tar'.format(0))


for epoch in range(args.epochs):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, features = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Class selective regularizer
        class_activations = {
            1: {},
            2: {},
            3: {},
            4: {}, 
            5: {}, 
            6: {}
        }    

        class_selectivity = {
            1: {},
            2: {},
            3: {},
            4: {}, 
            5: {}, 
            6: {}
        }      
    

        targets = labels.cpu().numpy().tolist()
        class_activations = forward_grad(features, targets, class_activations, args.return_nodes)

    # Layer_k = outer layer num, layer_v = dict of the form {class_i: {} ... } 
        for layer_k, layer_v in class_activations.items():
            # for class_k, class_v in class_activations[layer_k].items():
            # For a layer, the number of bottleneck layers will be the same 
            # So, just choose any class to get the index of bottleneck layers 
            random_key = targets[0]

            for bottleneck_k, bottleneck_v in class_activations[layer_k][random_key].items():
                for ci, class_k in enumerate(sorted(class_activations[layer_k].keys())):
                    if ci > 0:
                        all_activations_for_this_bottleneck = torch.cat((all_activations_for_this_bottleneck, class_activations[layer_k][class_k][bottleneck_k]), dim=0)
                    else:
                        all_activations_for_this_bottleneck = class_activations[layer_k][class_k][bottleneck_k]
                
                all_activations_for_this_bottleneck = all_activations_for_this_bottleneck.t()

                u_max, u_max_indices = torch.max(all_activations_for_this_bottleneck, dim=1)
                u_sum = torch.sum(all_activations_for_this_bottleneck, dim=1)
                u_minus_max = (u_sum - u_max) / (all_activations_for_this_bottleneck.shape[1] - 1)

                selectivity = (u_max - u_minus_max) / (u_max + u_minus_max + epsilon)
                
                class_selectivity[layer_k].update({bottleneck_k: selectivity})
    
        layer_selectivity = []
        for layer_k, layer_v in class_selectivity.items():
            unit_selectivity = []
            if layer_k not in (5, 6): 
                for bottleneck_k, bottleneck_v in class_selectivity[layer_k].items():
                    unit_selectivity += class_selectivity[layer_k][bottleneck_k]
                avg_unit_selectivity = sum(unit_selectivity) / len(unit_selectivity)
                layer_selectivity.append(avg_unit_selectivity)
 
        regularization_term = sum(layer_selectivity) / len(layer_selectivity)

        alpha = args.alphas[epoch]
        loss = criterion(outputs, labels) - alpha*regularization_term
        # loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()


    train_acc1 = 100*(correct / total) 
    acc1 = validate(model)

    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if epoch % 5 == 0:    # print every 2000 mini-batches
        print(f'[{epoch + 1}] loss: {running_loss / 10:.3f}, train_acc: {train_acc1}, val_acc: {acc1}, best: {best_acc1}')
        running_loss = 0.0

    class_selectivity, class_activations = get_class_selectivity_cus_model(checkpoint=model, val_loader=testloader) 
    utils.save_file(class_selectivity, EXP_DIR / 'cs_dict_{}_cp{}'.format(loader, epoch+1))
    utils.save_file(class_activations, EXP_DIR / 'cs_dict_{}_cp{}_full'.format(loader, epoch+1))


    save_checkpoint({
            'epoch': epoch + 1,
            'arc': args.arc,
            #'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'val_acc1': acc1, 
            'train_acc1': train_acc1,
            #'optimizer' : optimizer.state_dict(),
        }, is_best, filename='checkpoint_e{}.pth.tar'.format(epoch+1))


print('Finished Training')


