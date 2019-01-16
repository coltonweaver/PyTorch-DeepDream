import torch
from torchvision import models, transforms

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops

# Using the VGG19 model trained on ImageNet, need only the layers
vgg = models.vgg19(pretrained=True).features

# Freeze the parameters because no need to update model
for param in vgg.parameters():
    param.requires_grad_(False)

# Move VGG model to GPU if available for faster processing.
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
vgg.to(device)

def transform_image(image, max_size=400, shape=None):
    '''
        Take an image and transform it to put it through algorithm.
    '''

    # Now process the input image
    image_trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])

    # Remove hidden alpha channel, keeping just RGB and unsqueeze to flatten it.
    image = image_trans(image)[:3,:,:].unsqueeze(0).to(device)
    return image

def im_convert(tensor):
    '''
        Takes in a tensor representing an image, and converts it to a proper Image
    '''
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0,1)

    return Image.fromarray(np.uint8(image * 255))

def image_forward(input_tensor, layer, iterations, lr):
    '''
        Runs the given image through the model to the given layer for the given number of iterations 
    '''
    # Begin iterations with given image
    for _ in range(iterations):
        temp_tensor = input_tensor
        temp_tensor.requires_grad_(True)

        # Input image through vgg model to specified layer
        i = 0
        for _, layer in vgg._modules.items():
            if i == iterations:
                break
            temp_tensor = layer(temp_tensor)
            i += 1

        
        # Now track the loss and grad at the layer to apply to input image
        loss = temp_tensor.norm()
        loss.backward()

        # Add the learned gradient data to the input image data
        input_tensor.data = input_tensor.data + lr * input_tensor.grad.data
    
    return im_convert(input_tensor)

def deep_dream(image, layer, iterations, lr, octave_scale, num_octaves):
    '''
        Takes the image, requested layer, iterations, learning rate, and scale and performs the deep dream transformation
    '''
    if num_octaves > 0:
        temp_image = image

        # Downscale the image by octave_scale
        if (temp_image.size[0] / octave_scale < 1) or (temp_image.size[1] / octave_scale < 1):
            size = temp_image.size
        else:
            size = (int(temp_image.size[0]/octave_scale), int(temp_image.size[1]/octave_scale))

        # Call this function recursively to apply the deep dream filter to downscaled versions of this image
        temp_image = temp_image.resize(size, Image.ANTIALIAS)
        temp_image = deep_dream(temp_image, layer, iterations, lr, octave_scale, num_octaves - 1)

        temp_image = temp_image.resize(image.size, Image.ANTIALIAS)

        # Blend the processed image into the original image
        image = ImageChops.blend(image, temp_image, 0.6)
    
    # This is where the deep dream logic is applied to this version of the image
    result = image_forward(transform_image(image), layer, iterations, lr)
    result = result.resize(image.size)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Takes a given image and applies DeepDream algorithm to it.")
    parser.add_argument('-f', dest='file', action='store', required=True, default=None, help='Path to input image.')
    parser.add_argument('-l', dest='layer', action='store', default=10, help='Desired network layer to push image through to. (Default = 10)')
    parser.add_argument('-i', dest='iter', action='store', default=11, help='Desired iterations to apply to image. (Default = 11)')
    parser.add_argument('-lr', dest='lr', action='store', default=0.01, help='Desired learning rate to use in algorithm. (Default = 0.01)')

    args = parser.parse_args()

    input_image = Image.open(args.file)
    result = deep_dream(input_image, args.layer, args.iter, args.lr, 2, 20)

    plt.imshow(result)
    plt.show()

    result.save('out.png')
