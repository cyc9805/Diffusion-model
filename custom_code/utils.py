import torchvision
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def load_data(image_size, batch_size=5, dataset_name=None):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor()
    ])

    if dataset_name == 'CelebA':
        dataset = torchvision.datasets.CelebA(root='/home/cyc/Diffusion_model/running_with_diffusers/dataset', download=True, transform=transforms)
    
    elif dataset_name == 'Stanford_cars':
        dataset = torchvision.datasets.stanford_cars(root='/home/cyc/Diffusion_model/running_with_diffusers/dataset', download=True, transform=transforms)

    elif dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='/home/cyc/Diffusion_model/running_with_diffusers/dataset', download=True, transform=transforms)

    else:
        raise Exception('dataset name is not valid')

    # for custom dataset
    # dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'successfully loaded dataset {dataset_name}!')
    return dataloader

def custom_imshow(img):
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

