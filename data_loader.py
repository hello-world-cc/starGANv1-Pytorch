import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    def __init__(self, transform,img_path,attr_path,mode):
        super(MyDataset, self).__init__()
        self.transform = transform
        self.img_path = img_path
        self.attr_path = attr_path
        self.f = open(attr_path, 'r')
        self.mode = mode
        self.list = self.f.readlines()

    def __getitem__(self, index):
        index = index+1
        if self.mode == 'train':
            index = index+2000
        img_name = str(index).zfill(6) + ".jpg"
        img = Image.open(self.img_path + img_name)
        label = np.zeros(5)
        if self.list[index + 1].rstrip('\n').split()[9] == "1":
            label[0] = 1
        if self.list[index + 1].rstrip('\n').split()[10] == "1":
            label[1] = 1
        if self.list[index + 1].rstrip('\n').split()[12] == "1":
            label[2] = 1
        if self.list[index + 1].rstrip('\n').split()[21] == "1":
            label[3] = 1
        if self.list[index + 1].rstrip('\n').split()[40] == "1":
            label[4] = 1
        img = self.transform(img)
        return img, torch.tensor(label,dtype=torch.float)

    def __len__(self):
        if self.mode == 'train':
            return len(self.list) - 3000
        else:
            return 2000

def get_loader(img_path,attr_path,mode,batch_size,crop_size,img_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = MyDataset(transform,img_path,attr_path,mode)
    return torch.utils.data.DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)