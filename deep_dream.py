import torch
import argparse
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops

# Determine if cuda is available on system
cuda_available = torch.cuda.is_available

# Using the VGG16 model trained on ImageNet
vgg = models.vgg16(pretrained=True)

# Move the model to GPU memory if available
if cuda_available:
    vgg = vgg.cuda()

# Extract the exact layers to access them individually
layer_list = list(vgg.features.children())

# Normalize the input to the VGG16 network
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# All PyTorch Models require resizing to 224x224, converted to tensor, and normalized
process_image = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
])

# Function takes image tensor, removes normalization values and returns it
def deprocess(image_tensor):
    std_tensor = torch.Tensor([0.229, 0.224, 0.225])
    mean_tensor = torch.Tensor([0.485, 0.456, 0.406])

    if cuda_available:
        std_tensor, mean_tensor = std_tensor.cuda(), mean_tensor.cuda()

    return image_tensor * std_tensor + mean_tensor

# Function puts given image through model to specified layer for given iterations
def helper(raw_image, layer, iterations, lr):
    input_tensor = process_image(raw_image).unsqueeze(0)

    if cuda_available:
        input_tensor = input_tensor.cuda()

    # Zero out all gradients of the model parameters
    vgg.zero_grad()

    # Begin iterations with given image
    for _ in range(iterations):
        temp_tensor = input_tensor
        temp_tensor.requires_grad_()

        # Input image through vgg model to specified layer
        for j in range(layer):
            temp_tensor = layer_list[j](temp_tensor)
        
        # Now track the loss and grad at the layer to apply to input image
        loss = temp_tensor.norm()
        loss.backward()

        # Add the learned gradient data to the input image data
        input_tensor.data = input_tensor.data + lr * input_tensor.grad.data

    # Take processed tensor and return to image type
    input_tensor = input_tensor.data.squeeze()
    input_tensor.transpose_(0,1)
    input_tensor.transpose_(1,2)
    input_tensor = deprocess(input_tensor)

    return Image.fromarray(np.uint8(input_tensor.cpu() * 255))

# Handles processing the image through the model
def deep_dream(raw_image, layer, iterations, lr, octave_scale, num_octaves):

    if num_octaves > 0:
        temp_image = raw_image

        # Downscale the image by octave_scale
        if (temp_image.size[0] / octave_scale < 1) or (temp_image.size[1] / octave_scale < 1):
            size = temp_image.size
        else:
            size = (int(temp_image.size[0]/octave_scale), int(temp_image.size[1]/octave_scale))

        # Call this function recursively to apply the deep dream filter to downscaled versions of this image
        temp_image = temp_image.resize(size, Image.ANTIALIAS)
        temp_image = deep_dream(temp_image, layer, iterations, lr, octave_scale, num_octaves - 1)

        temp_image = temp_image.resize(raw_image.size, Image.ANTIALIAS)

        # Blend the processed image into the original image
        raw_image = ImageChops.blend(raw_image, temp_image, 0.85)
    
    # This is where the deep dream logic is applied to this version of the image
    result = helper(raw_image, layer, iterations, lr)
    result = result.resize(raw_image.size)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Takes a given image and applies DeepDream algorithm to it.")
    parser.add_argument('-f', dest='file', action='store', required=True, default=None, help='Path to input image.')
    parser.add_argument('-l', dest='layer', action='store', default=16, help='Desired network layer to push image through to. (Default = 16)')
    parser.add_argument('-i', dest='iter', action='store', default=11, help='Desired iterations to apply to image. (Default = 11)')
    parser.add_argument('-lr', dest='lr', action='store', default=0.0012, help='Desired learning rate to use in algorithm. (Default = 0.0012)')

    args = parser.parse_args()

    raw_image = Image.open(args.file)
    result = deep_dream(raw_image, args.layer, args.iter, args.lr, 2, 20)

    plt.imshow(result)
    plt.show()

    result.save('out.png')