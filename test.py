import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
import tqdm
import os
import time
from torch.nn import init
from arch import generator
from arch import discriminator
import utils
import data_loader
import itertools

class starGAN(nn.Module):
    def __init__(self,args):
        super(starGAN, self).__init__()
        self.device = torch.device("cuda:"+str(args.cuda_id)+"" if torch.cuda.is_available() else "cpu")
        self.G = generator.Generator().to(self.device)
        self.D = discriminator.Discriminator().to(self.device)
        utils.print_networks([self.G, self.D], ['G', 'D'])
        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.model_save_dir))
            self.G.load_state_dict(ckpt['G'])
            self.D.load_state_dict(ckpt['D'])
        except:
            print(' [*] No checkpoint!')
        self.test_loader = data_loader.get_loader(args.img_path, args.attr_path, args.mode, args.batch_size, args.crop_size,
                                              args.img_size)


    def test(self,args):
        for i, (x, y) in enumerate(self.test_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            for j, (target_x, target_y) in enumerate(self.test_loader):
                target_x = target_x.to(self.device)
                target_y = target_y.to(self.device)
                break
            with torch.no_grad():
                self.G.eval()
                y_ = y
                y_[:, 0:3] = torch.FloatTensor([1, 0, 0])
                test_images_balck_hair = self.G(x, y_)
                y_ = y
                y_[:, 0:3] = torch.FloatTensor([0, 1, 0])
                test_images_blond_hair = self.G(x, y_)
                y_ = y
                y_[:, 0:3] = torch.FloatTensor([0, 0, 1])
                test_images_brown_hair = self.G(x, y_)
                y_ = y
                y_[:, 3] = torch.FloatTensor([1])
                test_images_man = self.G(x, y_)
                y_ = y
                y_[:, 4] = torch.FloatTensor([1])
                test_images_young = self.G(x, y_)
                test_images = self.G(x, target_y)

            size_figure_grid = 6
            fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(10, 10))
            for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)
            ax[0, 0].cla()
            ax[0, 0].imshow(np.transpose(x[0].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))
            ax[0, 1].cla()
            ax[0, 1].imshow(np.transpose(test_images_balck_hair[0].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))
            ax[0, 2].cla()
            ax[0, 2].imshow(np.transpose(test_images_blond_hair[0].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))
            ax[0, 3].cla()
            ax[0, 3].imshow(np.transpose(test_images_brown_hair[0].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))
            ax[0, 4].cla()
            ax[0, 4].imshow(np.transpose(test_images_man[0].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))
            ax[0, 5].cla()
            ax[0, 5].imshow(np.transpose(test_images_young[0].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))
            for k in range(1, 6):
                ax[k, 0].cla()
                ax[k, 0].imshow(np.transpose(x[2 * k].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))
                ax[k, 1].cla()
                ax[k, 1].imshow(np.transpose(target_x[2 * k].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))
                ax[k, 2].cla()
                ax[k, 2].imshow(np.transpose((test_images)[2 * k].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))

                ax[k, 3].cla()
                ax[k, 3].imshow(np.transpose(x[2 * k + 1].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))
                ax[k, 4].cla()
                ax[k, 4].imshow(np.transpose(target_x[2 * k + 1].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))
                ax[k, 5].cla()
                ax[k, 5].imshow(np.transpose((test_images)[2 * k + 1].cpu().data.numpy() * 0.5 + 0.5, (1, 2, 0)))
            plt.show()
            break


