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
from class_selectivity_v3 import get_class_selectivity
import utils 


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
            outputs = model(images)
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
args = parser.parse_args()

EXP_DIR = DATA_PATH / args.exp_name
os.makedirs(EXP_DIR, exist_ok=True)
args.arc = 'vgg16' 
loader = 'val'

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


model = models.vgg16()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_acc1 = 0
acc1 = validate(model)

class_selectivity, class_activations = get_class_selectivity(model=model, val_loader=testloader) 
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
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
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

    class_selectivity, class_activations = get_class_selectivity(model=model, val_loader=testloader) 
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


