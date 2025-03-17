import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# Set device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None, max_size=None, shape=None):
    """Load an image and convert it to a torch tensor."""
    image = Image.open(image_path)

    if max_size:
        # Compute the scaling factor while preserving the aspect ratio
        original_size = np.array(image.size)
        scale = max_size / max(original_size)
        new_size = tuple((original_size * scale).astype(int))  # Ensure it's a tuple of integers
        image = image.resize(new_size, Image.LANCZOS)

    if shape:
        image = image.resize(tuple(shape), Image.LANCZOS)

    if transform:
        image = transform(image).unsqueeze(0)

    return image.to(device)  # Move image tensor to the selected device

class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

def main(config):

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # Load content and style images, and move them to device
    content = load_image(config['content_path'], transform, max_size=config['max_size'])
    style = load_image(config['style_path'], transform, shape=[content.size(2), content.size(3)])

    # Initialize target image with content image and move to device
    target = content.clone().requires_grad_(True).to(device)

    optimizer = torch.optim.Adam([target], lr=config['lr'], betas=(0.5, 0.999))
    vgg = VGGNet().to(device).eval()

    for step in range(config['total_step']):

        # Extract multiple (5) conv feature vectors
        target_features = vgg(target)
        content_features = vgg(content)
        style_features = vgg(style)

        style_loss = 0
        content_loss = 0
        for f1, f2, f3 in zip(target_features, content_features, style_features):
            # Compute content loss with target and content images
            content_loss += torch.mean((f1 - f2)**2)

            # Reshape convolutional feature maps for style loss computation
            _, c, h, w = f1.size()
            f1 = f1.view(c, h * w)
            f3 = f3.view(c, h * w)

            # Compute Gram matrix
            f1_gram = torch.mm(f1, f1.t())
            f3_gram = torch.mm(f3, f3.t())

            # Compute style loss with target and style images
            style_loss += torch.mean((f1_gram - f3_gram)**2) / (c * h * w)

        # Compute total loss, backprop and optimize
        loss = content_loss + config['style_weight'] * style_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % config['log_step'] == 0:
            print(f'Step [{step + 1}/{config["total_step"]}], Content Loss: {content_loss.item():.4f}, Style Loss: {style_loss.item():.4f}')

        if (step + 1) % config['sample_step'] == 0:
            # Save the generated image
            denorm = transforms.Normalize(mean=(-2.12, -2.04, -1.80), std=(4.37, 4.46, 4.44))
            img = target.clone().squeeze()
            img = denorm(img).clamp_(0, 1)  # Clamping the values between 0 and 1 for valid pixel values
            torchvision.utils.save_image(img, os.path.join(config['output_dir'], f'output-{step + 1}.png'))

# Configuration
config = {
    'content_path': "/scratch/ep23btech11012.phy.iith/style_transfer/content_images/WhatsApp Image 2024-10-28 at 16.33.45.jpeg",
    'style_path': "/scratch/ep23btech11012.phy.iith/style_transfer/style_images/2b6c6aa85feeb31a053bcf6ff10852a5.jpg",
    'output_dir': '/scratch/ep23btech11012.phy.iith/style_transfer/output5/',
    'max_size': 400,
    'total_step': 10000,
    'log_step': 10,
    'sample_step': 250,
    'style_weight': 100,
    'lr': 0.003
}

# Create output directory if it doesn't exist
if not os.path.exists(config['output_dir']):
    os.makedirs(config['output_dir'])

# Run the main function
main(config)
