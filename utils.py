import os
import torch

def print_networks(nets, names):
    print('------------Number of Parameters---------------')
    i = 0
    for net in nets:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network %s] Total number of parameters : %.3f M' % (names[i], num_params / 1e6))
        i = i+1
    print('-----------------------------------------------')


def save_checkpoint(state, save_path):
    torch.save(state, save_path)


# To load the checkpoint
def load_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path)
    print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt

# To make directories
def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)












