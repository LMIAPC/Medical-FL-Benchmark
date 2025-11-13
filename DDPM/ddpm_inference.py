import os
import math
import torch
import cv2
import numpy as np
from tqdm import tqdm
from utils import *
from modules import UNet_conditional
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, ch, labels, scale):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, ch, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        for i in range (x.shape[0]):
            y = 0.5*(x[i] + 1)
            y = y.clamp(0,1)
            x[i] = (y * 255)
            x[i] *= scale

        return x.type(torch.uint8)

def test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('--image_size', type=int, default=256, help='image_size')
    parser.add_argument('--channels', type=int, default=3, help='channels')
    parser.add_argument('--round', type=int, default=3400, help='round of model used')
    parser.add_argument('--num', type=int, default=560, help='generate data number')
    args = parser.parse_args()

    device = "cuda"
    ch = args.channels

    round = args.round
    gen_num = args.num
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    model_name = f"./models/{args.dataset}/ema_ckpt_{round}.pt"
    ckpt = torch.load(model_name)
    print('load', model_name)
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    for k in range (0, args.num_classes):
        cls = k
        savePath = os.path.join(args.output, args.dataset, str(cls)+'_'+str(cls))
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        
        print('generate class ', cls)
        nums = 16
        rounds = math.ceil(float(gen_num/args.num_classes)/nums)
        for idx in range(rounds):
            scale = 1.0
            y = torch.Tensor([cls] * nums).long().to(device)
            sampled_images = diffusion.sample(model, nums, ch, y, scale).squeeze().to('cpu').numpy()
            if (ch > 1): 
                sampled_images = np.transpose(sampled_images, [0, 2, 3, 1])

            for i in range (sampled_images.shape[0]):
                name = os.path.join(savePath, f"{idx}_{i}.png")
                img = cv2.cvtColor(sampled_images[i],cv2.COLOR_BGR2RGB)
                cv2.imwrite(name, img)

if __name__ == '__main__':
    test()