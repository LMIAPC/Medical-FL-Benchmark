import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from .ddpm_util import *
from modules import UNet_conditional, EMA
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def lr_lambda(epoch):
    warm_up_epoch = 20 
    lr_max = 0.01
    lr_min = 3e-4
    if epoch <= warm_up_epoch:
        lr = max((epoch / warm_up_epoch) * lr_max, lr_min)
    else:
        lr = max(lr_min, lr_max * 0.9**(epoch - warm_up_epoch))
    lr_rate = lr / lr_min
    return lr_rate

class Diffusion:
    def __init__(self, noise_steps=10000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def train(args):
    start_epoch = 0
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    if args.flag:
        model.load_state_dict(torch.load(f"models/{args.run_name}/ckpt_{start_epoch}.pt"))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.flag:
        optimizer.load_state_dict(torch.load(f"models/{args.run_name}/optim_{start_epoch}.pt"))
    if args.flag:
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=start_epoch)
    else:
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    ema = EMA(0.995)

    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    if args.flag:
        ema_model.load_state_dict(torch.load(f"models/{args.run_name}/ema_ckpt_latest.pt"))
        ema.step += start_epoch
    
    
    dlen = len(dataloader)
    epochs = int(1e6 * args.num_classes / (dlen*args.batch_size))
    epochs = math.ceil(epochs / 100) * 100 + 1
    print('dlen, epochs:', dlen, epochs)
    model.train()
    ema_model.eval()
    for epoch in range(start_epoch, epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())

        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt_latest.pt"))
        torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt_latest.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim_latest.pt"))

        if epoch % 100 == 0:
            labels = torch.arange(args.num_classes).long().to(device)
            ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.png"))
            torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt_{epoch}.pt"))
            torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt_{epoch}.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim_{epoch}.pt"))
    

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--dataset_path', type=str, default='./', help='dataset root dir')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size root dir')
    parser.add_argument('--num_classes', type=int, default=10, help='number of labeles')
    parser.add_argument('--image_size', type=int, default=64, help='image_size')
    parser.add_argument('--channels', type=int, default=3, help='channels')
    parser.add_argument('--flag', action='store_true', help='use saved model parameters')

    args = parser.parse_args()
    args.run_name = args.dataset + '_' + os.path.basename(args.dataset_path)

    args.device = "cuda"
    args.lr = 3e-4
    train(args)
    

if __name__ == '__main__':
    launch()

