from __future__ import print_function
import argparse
import os

import torchvision.transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from skimage import io
import torch
from model import *
import cv2
from DataLoader_duke17 import *

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--threads", type=int, default=0, help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

generator =KGenerator()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.cuda.set_device(0)

if cuda:
    generator = generator.cuda()

print('===> Loading datasets')

val_dataloader = octDataset('duke17.csv')
validation_data_loader = DataLoader(dataset=val_dataloader, num_workers=opt.threads, batch_size=opt.batch_size,
                                    shuffle=False)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def inference():
    generator.load_state_dict(torch.load(r"generator.pth", map_location='cpu'))
    root_result_test = r"sna-skan/saved_models_all_kan_duke17"
    os.makedirs(root_result_test,exist_ok=True)
    index = 0

    for i, batch in enumerate(validation_data_loader):
        index = index + 1

        _,_,h,w=batch["A"].shape
        noisy = (Variable(batch["A"]))
        clean = (Variable(batch["B"]))
        if cuda:
            noisy = noisy.cuda()
        with torch.no_grad():
            # print(noisy.shape)
            noise = generator(noisy)
            imgout_test = noisy - noise
            imgout_test = (imgout_test[0][0]).detach().cpu().numpy()


            imgout_test[imgout_test > 1.0] = 1.0
            imgout_test[imgout_test < 0.0] = 0.0

            cimg = clean.squeeze().squeeze().cpu().detach().numpy()
            print(imgout_test.shape)
            print(cimg.shape)

        imgout_test = imgout_test * 255.0
        clean=cimg*255

        print(imgout_test)
        # cv2.imshow(np.array(imgout_test))
        filename_result_test = str(index) + '_noise_diff.png'
        filename_abs_root_test = os.path.join(root_result_test, filename_result_test)
        cv2.imwrite(filename_abs_root_test, imgout_test)



inference()
