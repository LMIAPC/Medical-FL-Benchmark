import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    if args.dataset_path.find('TB') >= 0:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.9, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif args.dataset_path.find('Retino') >= 0:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.9, 1.0)), 
            torchvision.transforms.RandomHorizontalFlip(p=0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.9, 1.0)), 
            torchvision.transforms.RandomHorizontalFlip(p=0.1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    print(dataset.class_to_idx)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def model_init(model_weight, model_bias=None):
    if not model_weight:
        torch.nn.init.xavier_normal(model_weight)
    if not model_bias:
        torch.nn.init.zeros_(model_bias)
