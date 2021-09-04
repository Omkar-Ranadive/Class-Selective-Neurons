import torch
from torch import nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from constants import DATA_PATH
import numpy as np 


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


def predict(model, batch): 

    with torch.no_grad():
        output = model(batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        # print(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        print("Number of output probs (classes): {}".format(len(probabilities)))

    return probabilities

def get_topk(probabilities, k=5):
    """
    Get the top k categories predicted by the model 
    Args:
        probabilities (tensor): Probabilities tensor outputted by the model (predict function)
        k (int, optional): (Number of categories). Defaults to 5.
    """

    # Read the categories
    with open(DATA_PATH / "imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    topk_prob, topk_catid = torch.topk(probabilities, k)
    print("Top {} categories are: ".format(k))
    for i in range(topk_prob.size(0)):
        print(categories[topk_catid[i]], topk_prob[i].item())


def ablate(state_dict, keys, num_channels=100): 
    
    for k in keys: 
        arr = state_dict[k]
        # Note: Weight matrices are shaped as follows in the Resnet model 
        '''
        Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        torch.Size([512, 1024, 1, 1])

        Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        torch.Size([256, 256, 3, 3])

        Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        torch.Size([512, 2048, 1, 1])
        '''

        # Randomly ablate output channels 
        channels = np.random.permutation(arr.shape[1])
        # Set those random channels to zero 
        arr[:, channels[:num_channels], :, :] = 0 

        # Update the state dict 
        state_dict[k] = arr 

    return state_dict


def ablate_using_activations(model, input_batch, key, num_channels=100):
    resnet_layers = nn.Sequential(*list(model.children())[:9])
    for index, intermediate_model in enumerate(resnet_layers):
        if index == key:
            # Randomly ablate output channels 
            channels = np.random.permutation(input_batch.shape[1])
            # Set those random channels to zero 
            input_batch[:, channels[:num_channels], :, :] = 0
        input_batch = intermediate_model(input_batch)
    input_batch = intermediate_model(input_batch)
    input_batch = intermediate_model(input_batch)

    return input_batch


if __name__ == '__main__': 
    # Load pre-trained Resnet 
    model = models.resnet50(pretrained=True)
    model.eval()
    

    # Load the data 
    input_image = Image.open(DATA_PATH / 'dog.jpg')
    input_tensor = process_img(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    probs = predict(model, input_batch)
    get_topk(probs)
    print("------------------------------------------------------")

    state_dict = model.state_dict()
    # print(state_dict.keys())
    # print(state_dict['layer3.4.conv2.weight'].shape, state_dict['layer3.4.bn2.weight'].shape)

    # layers = dict(model.named_modules())
    # print(*list(model.children()))
    # print(nn.ModuleList(list(model.children())[:23]))
    # print(layers['layer4.0.conv1'])
    # print(state_dict['layer4.0.conv1.weight'].shape)
    # print(layers['layer3.0.conv2'])
    # print(state_dict['layer3.0.conv2.weight'].shape)
    # print(layers['layer4.2.conv1'])
    # print(state_dict['layer4.2.conv1.weight'].shape)

    # ablate(state_dict, keys=['layer4.0.conv1.weight'])

    # Ablate values 
    # state_dict_updated = ablate(state_dict, keys=['layer4.0.conv1.weight'])
    # model.load_state_dict(state_dict_updated)

    # Ablate values using activation
    # Returns output for 2nd last layer instead of last layer. So it contains 2048 neurons instead of 1000.
    #TODO: Address this issue. Get output for last layer.
    output = ablate_using_activations(model, input_batch, 6)
    # print("Output shape:", output.size())
    probs_with_ablation = torch.nn.functional.softmax(output[0], dim=0).squeeze(1).squeeze(1)
    # print("Probs shape:", probs_with_ablation.size())
    get_topk(probs_with_ablation)

    # Predict again 
    # probs = predict(model, input_batch)
    # print(probs.size())
    # get_topk(probs)

