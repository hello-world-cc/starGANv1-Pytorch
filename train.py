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

class starGAN(nn.Module):
    def __init__(self,args):
        super(starGAN, self).__init__()
        self.device = torch.device("cuda:"+str(args.cuda_id)+"" if torch.cuda.is_available() else "cpu")
        self.G = generator.Generator().to(self.device)
        self.D = discriminator.Discriminator().to(self.device)
        # self.init_weights(self.G)
        # self.init_weights(self.D)
        utils.print_networks([self.G, self.D], ['G', 'D'])

        self.optim_G = torch.optim.Adam(params=self.G.parameters(),lr=args.g_lr,betas=(args.beta1,args.beta2))
        self.optim_D = torch.optim.Adam(params=self.D.parameters(), lr=args.d_lr, betas=(args.beta1,args.beta2))
        self.BCE = nn.BCELoss()
        self.L1 = nn.L1Loss()

        self.train_loader = data_loader.get_loader(args.img_path, args.attr_path, args.mode, args.batch_size, args.crop_size,
                                              args.img_size)


    def init_weights(net, gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.normal(m.weight.data, 0.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal(m.weight.data, 1.0, gain)
                init.constant(m.bias.data, 0.0)
        print('Network initialized with weights sampled from N(0,0.02).')
        net.apply(init_func)
    def train_D(self,args):
        D_loss = []
        for i, (x, y) in tqdm.tqdm(enumerate(self.train_loader)):
            x = x.to(self.device)
            y = y.to(self.device)
            rand_index = torch.randperm(y.size(0))
            random_label = y[rand_index].to(self.device)
            # random_label = torch.randint(0,2,size=(x.size(0),5)).float().to(device)
            fake = torch.zeros(x.size(0),1,2,2).to(self.device)
            valid = torch.ones(x.size(0),1,2,2).to(self.device)

            fake_img = self.G(x,random_label).detach()
            fake_src,fake_cls = self.D(fake_img) ###B*1*2*2   B*5
            # loss1 = criterion1(fake_src,fake)
            loss1 = torch.mean(fake_src)
            real_src,real_cls = self.D(x)
            # loss2 = criterion1(real_src,valid)
            loss2 = -torch.mean(real_src)
            # loss3 =0
            # for j in range(5):
            #     loss3 = loss3+criterion1(real_cls[:,j],y[:,j])
            loss3 = self.BCE(real_cls,y)
            alpha = torch.rand(x.size(0),3,128,128).to(self.device)  #添加梯度惩罚项
            x_ = alpha*x+(1-alpha)*fake_img
            x_.requires_grad_(True)
            y_, _ = self.D(x_)
            gradients = torch.autograd.grad(outputs=y_, inputs=x_,
                                            grad_outputs=torch.ones(y_.size()).to(self.device),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0),-1)
            gradient_penalty = torch.pow((torch.sum(gradients**2, dim=1) - 1), 2).mean()
            loss_D = loss1+loss2+loss3*args.lambda_cls+gradient_penalty*args.lambda_gp
            D_loss.append(float(loss_D.item()))
            self.optim_D.zero_grad()
            loss_D.backward()
            self.optim_D.step()
            return D_loss
    def train_G(self,args):
        G_loss = []
        for i, (x, y) in tqdm.tqdm(enumerate(self.train_loader)):
            x = x.to(self.device)
            y = y.to(self.device)
            rand_index = torch.randperm(y.size(0))
            random_label = y[rand_index].to(self.device)
            # random_label = torch.randint(0,2,size=(x.size(0),5)).float().to(device)
            fake = torch.zeros(x.size(0),1,2,2).to(self.device)
            valid = torch.ones(x.size(0),1,2,2).to(self.device)
            fake_img = self.G(x, random_label)
            fake_src, fake_cls = self.D(fake_img)  ###B*1*2*2   B*5
            # loss1 = criterion1(fake_src, valid)
            loss1 = -torch.mean(fake_src)
            # loss2 = 0
            # for j in range(5):
            #     loss2 = loss2+criterion1(fake_cls[:,j], random_label[:,j])
            loss2 = self.BCE(fake_cls,random_label)
            loss3 = self.L1(x, self.G(fake_img, y))
            loss_G = loss1 + args.lambda_cls*loss2 + loss3*args.lambda_rec
            G_loss.append(float(loss_G.item()))
            self.optim_G.zero_grad()
            loss_G.backward()
            self.optim_G.step()
            return G_loss

    def updata_lr(self):
        for param_group in self.optim_G.param_groups:
            param_group['lr'] -=param_group['lr']/10000
        for param_group in self.optim_D.param_groups:
            param_group['lr'] -=param_group['lr']/10000
    def train(self,args):

        for epoch in range(args.train_epoch):
            if (epoch + 1) % 1000 == 0:
                self.updata_lr()
            D_loss = []
            G_loss = []
            for j in range(args.n_critic):
                D_loss = self.train_D(args)
            G_loss = self.train_G(args)
            print("epoch:", epoch + 1, "D_loss:", torch.mean(torch.FloatTensor(D_loss)))
            print("epoch:", epoch + 1, "G_loss:", torch.mean(torch.FloatTensor(G_loss)))
            if (epoch+1)%args.model_save_epoch == 0:
                utils.mkdir(args.model_save_dir)
                utils.save_checkpoint({'epoch': epoch + 1,
                                       'D': self.D.state_dict(),
                                       'G': self.G.state_dict()},
                                      '%s/latest.ckpt' % (args.model_save_dir))
                # utils.save_checkpoint(self.G.state_dict(), os.path.join(args.model_save_dir, 'G_' + str(epoch + 1) + '.pkl'))
                # utils.save_checkpoint(self.D.state_dict(), os.path.join(args.model_save_dir, 'D_' + str(epoch + 1) + '.pkl'))

























