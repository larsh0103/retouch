import argparse
import os
import numpy as np
import math
from torch.nn.modules.dropout import FeatureAlphaDropout
from torchvision.models.vgg import VGG

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import torch.nn as nn
import torch
import wandb
from model import Generator, Discriminator, GANLoss, VggShort
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--log_interval", type=int, default=10, help="log every n batches")
parser.add_argument("--rel_avg_gan", action="store_true", help="relativistic average GAN instead of standard")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Loss function
# adversarial_loss = torch.nn.BCEWithLogitsLoss().to(device)

config = wandb.config

config = vars(opt)

wandb.init(project="retouch",config=config)


criterionGAN = GANLoss('vanilla').to(device)
criterionMSE = nn.MSELoss()



# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Initialize vgg_classifier for feature based perceptual loss
VGG = VggShort().to(device)

dataset = utils.RetouchDataset(original_dir="ffhq",retouch_dir = "ffhqr", image_start = 0, image_stop = 9000,transform = utils.get_transforms(), target_transform=utils.get_target_transforms())
dataloader = DataLoader(dataset,batch_size=opt.batch_size,shuffle=False,num_workers=4)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
checkpoint_folder = "checkpoints"

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# # ----------
# #  Training
# # ----------

for epoch in range(opt.n_epochs):
    for i, (real_A, real_B) in enumerate(dataloader):

        # -----------------
        #  Train Generator
        # -----------------
        loss_G = torch.tensor([0.0]).to(device)
        loss_D = torch.tensor([0.0]).to(device)
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()

        # Generate fake B domain images from real A domain images
        fake_B = generator(real_A.to(device))
        # fake_AB = torch.cat((real_A.to(device),fake_B),1)
        # pred_fake = discriminator(fake_AB.detach())
        
        # real_AB = torch.cat((real_A.to(device),real_B.to(device)),1)
        # pred_real = discriminator(real_AB)
       
        pred_fake = discriminator(fake_B)
        pred_real = discriminator(real_B.to(device))

      
        features_fake = VGG(fake_B)
        features_real = VGG(real_B.to(device))

        # Loss measures generator's ability to fool the discriminator
        loss_G_gan = (criterionGAN(pred_fake- torch.mean(pred_real), False) + criterionGAN(pred_real- torch.mean(pred_fake), True)) / 2
        loss_G_mse =  criterionMSE(fake_B,real_B.to(device)) * 0.5
        loss_G_perceptual = criterionMSE(features_fake,features_real)

        loss_G += (10e-3 *loss_G_gan + 10e-3 *loss_G_perceptual + 10e-2 *loss_G_mse)
        
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

       # Generate fake B domain images from real A domain images
        fake_B = generator(real_A.to(device))
        
        # fake_AB = torch.cat((real_A.to(device),fake_B),1)
        # pred_fake = discriminator(fake_AB.detach())
        
        # real_AB = torch.cat((real_A.to(device),real_B.to(device)),1)
        # pred_real = discriminator(real_AB)

        pred_fake = discriminator(fake_B)
        pred_real = discriminator(real_B.to(device))

        loss_D_fake = criterionGAN(pred_fake- torch.mean(pred_real), False)
        loss_D_real = criterionGAN(pred_real- torch.mean(pred_fake), True)
        loss_D += ((loss_D_fake + loss_D_real) * 0.5)

        loss_D.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        )
        wandb.log({"D_batch_loss":loss_D.item(),"step": (epoch*len(dataloader) + i) })
        wandb.log({"G_batch_loss":loss_G.item(),"step": (epoch*len(dataloader) + i) })
        
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            wandb.log({"real_A":wandb.Image(real_A),"real_B": wandb.Image(real_B),"fake_B": wandb.Image(fake_B)})
            torch.save(generator.state_dict(),os.path.join(wandb.run.dir,f"generator_{(epoch*len(dataloader) + i)}.pth"))
            torch.save(discriminator.state_dict(),os.path.join(wandb.run.dir,f"discriminator_{(epoch*len(dataloader) + i)}.pth"))
