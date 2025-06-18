from __future__ import print_function
import argparse
import os
import random
from US_perceloss import PerceptualLoss
import numpy as np
import sys
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from model import *
from DataLoader_train import *


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--threads", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss
lambda_pixel = 6

# Calculate output of image discriminator (PatchGAN)
patch = (1, 40, 40)

# generator = Generator()
generator = KGenerator()
discriminator = Discriminator()
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
torch.cuda.set_device(1)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_pixelwise = criterion_pixelwise.cuda()
    p_loss=PerceptualLoss().cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0004, betas=(opt.b1, opt.b2))

print('===> Loading datasets')
dataloader = svd_get_Training_Set()
training_data_loader = DataLoader(dataset=dataloader, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def train(epoch):
    epoch_loss_D = 0
    epoch_loss_G = 0
    epoch_loss_Pixel = 0
    epoch_loss_GAN = 0
    for i, batch in enumerate(training_data_loader):
        I_A = Variable(batch["A"])
        I_harr = Variable(batch["B"])
        I_svd = Variable(batch["C"])
        C_C = Variable(batch["D"])

        if cuda:
            I_A = I_A.cuda()
            I_harr = I_harr.cuda()
            I_svd=I_svd.cuda()
            C_C = C_C.cuda()
        valid = torch.tensor(np.ones((I_A.size(0), *patch)), requires_grad=False).float().cuda()
        fake = torch.tensor(np.zeros((I_A.size(0), *patch)), requires_grad=False).float().cuda()

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_G.zero_grad()
        # GAN loss
        N__A = generator(I_A)

        C__A1 = I_A - N__A
        pred_C__A1 = discriminator(C__A1)
        pred_C__A1 = pred_C__A1.float()
        # print(pred_C__A1.shape,valid.shape)
        loss_GAN_C__A1 = criterion_GAN(pred_C__A1, valid)


        #定义基础噪声，预测噪声
        random_alpha=random.random()
        bais=0
        dot_alpha=random_alpha+bais
        base_noise=dot_alpha*I_harr+(1+bais-random_alpha)*I_svd
        #base_noise = 0.5 * I_harr + 0.5 * I_svd
        # base_noise = I_svd
        pred_noise=N__A
        I__A = C__A1 + base_noise

        C__A2 = I__A - generator(I__A)
        torch.cuda.empty_cache()
        pred_C__A2 = discriminator(C__A2)

        loss_GAN_C__A2 = criterion_GAN(pred_C__A2, valid)


        I__C1=C_C+base_noise
        I__C2 = C_C + pred_noise

        # I__C1 = C_C + N__A
        # I__C2 = C_C + N__B

        C__C1 = I__C1 - generator(I__C1)

        C__C2 = I__C2 - generator(I__C2)




        pred_C__C1 = discriminator(C__C1)
        pred_C__C2 = discriminator(C__C2)

        loss_GAN_C__C1 = criterion_GAN(pred_C__C1, valid)
        loss_GAN_C__C2 = criterion_GAN(pred_C__C2, valid)

        loss_GAN = loss_GAN_C__A1  + loss_GAN_C__A2 +  loss_GAN_C__C1 + loss_GAN_C__C2

        # identity loss
        loss_pixel_1 = criterion_pixelwise(C__A1, C__A2)
        loss_pixel_3 = criterion_pixelwise(C__C1, C_C)
        loss_pixel_4 = criterion_pixelwise(C__C2, C_C)

        # loss_pixel_p1,_ = p_loss(C__A1, C__A2)
        # loss_pixel_p2,_ = p_loss(C__C1, C_C)
        # loss_pixel_p3,_ = p_loss(C__C2, C_C)


        # loss_pixel_1 = criterion_pixelwise(C__A1, C__A2)
        # loss_pixel_2 = criterion_pixelwise(C__B1, C__B2)
        # loss_pixel_3 = criterion_pixelwise(C__C1, C_C)
        # loss_pixel_4 = criterion_pixelwise(C__C2, C_C)


        loss_pixel = loss_pixel_1 + (loss_pixel_3 + loss_pixel_4)*0.5


        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Real loss
        pred_C_C = discriminator(C_C)
        loss_real = criterion_GAN(pred_C_C, valid)

        # Fake loss 1
        pred_C__A1 = discriminator(C__A1.detach())

        loss_pred_C__A1 = criterion_GAN(pred_C__A1, fake)

        loss_fake_1 = loss_pred_C__A1

        # Fake loss 2
        pred_C__A2 = discriminator(C__A2.detach())

        loss_pred_C__A2 = criterion_GAN(pred_C__A2, fake)

        loss_fake_2 = loss_pred_C__A2

        # Fake loss 3
        pred_C__C1 = discriminator(C__C1.detach())
        pred_C__C2 = discriminator(C__C2.detach())
        loss_pred_C__C1 = criterion_GAN(pred_C__C1, fake)
        loss_pred_C__C2 = criterion_GAN(pred_C__C2, fake)
        loss_fake_3 = loss_pred_C__C1 + loss_pred_C__C2

        loss_fake = loss_fake_1 + loss_fake_2 + loss_fake_3

        # Total loss
        loss_D = loss_real + 0.1 * loss_fake
        loss_D.backward()
        optimizer_D.step()

        epoch_loss_D += loss_D.item()
        epoch_loss_G += loss_G.item()
        epoch_loss_Pixel += loss_pixel.item()
        epoch_loss_GAN += loss_GAN.item()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] "
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_pixel.item(),
                loss_GAN.item(),
            )
        )

    torch.save(generator.state_dict(), r"/sna-skan/saved_models_all_kan/generator_%d.pth" % (epoch))

os.makedirs('/home/ps/zhencunjiang/sna-skan/saved_models_all_kan',exist_ok=True)

for epoch in range(opt.epoch, opt.n_epochs):
    train(epoch)

# Save the final model checkpoints
torch.save(generator.state_dict(),
           r"/sna-skan/saved_models_all_kan/generator_%d.pth" % epoch)
torch.save(discriminator.state_dict(),
           r"/sna-skan/saved_models_all_kan/discriminator_%d.pth" % epoch)
