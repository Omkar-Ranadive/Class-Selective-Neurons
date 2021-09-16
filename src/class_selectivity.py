def get_activations(model):
    resnet_layers = nn.Sequential(*list(model.children()))

    for index, layer in enumerate(resnet_layers):

        if isinstance(layer, torch.nn.modules.linear.Linear): 
            # Flatten input batch once linear layer is reached 
            input_batch = torch.flatten(input_batch, start_dim=1)
        
        # If the present layer is sequential, go deeper 
        if isinstance(layer, torch.nn.modules.container.Sequential):
            
            for num, child in enumerate(layer.children()): 
                ablate = False 
                if isinstance(child, Bottleneck): 
                    # Note, except the first Relu, all Relu's are in the bottlneck layer 
                    # So, we only need to ablate activations in the bottleneck layer 
                    if index in keys and num in keys[index]: 
                        ablate = True 
                    input_batch = bottleneck_layer(input_batch, child, num_channels, ablate)
                else: 
                    input_batch = child(input_batch)

        else:
            input_batch = layer(input_batch)

    return input_batch


def bottleneck_layer(input_batch, child, num_channels, ablate):
    # The bottle neck layers also follow this structure 
    # Reference:  https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html 
    identity = input_batch

    out = child.conv1(input_batch)
    out = child.bn1(out)
    out = child.relu(out)
    
    if ablate: 
        out = zero_out_activation(out, num_channels)

    out = child.conv2(out)
    out = child.bn2(out)
    out = child.relu(out)

    out = child.conv3(out)
    out = child.bn3(out)

    if child.downsample is not None:
        identity = child.downsample(input_batch)

    out += identity
    out = child.relu(out) 

    if ablate: 
        out = zero_out_activation(out, num_channels)

    return out


def get_class_selectivity(model):
    # switch to evaluate mode
    model.eval()
    model.to('cuda')

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
        
            if torch.cuda.is_available():
                target = target.to('cuda')
                images = images.to('cuda')

            # compute output
            output = model(images)
            loss = criterion(output, target)

    
    