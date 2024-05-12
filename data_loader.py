import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from PIL import Image
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, RandomCrop, Resize
from torchvision.transforms.functional import hflip, rotate, crop
import glob
from torchvision.transforms import ToTensor, RandomCrop, Resize





class PairLoader(Dataset):
    def __init__(self,):
        self.filesClear = os.listdir(r"D:\clear_images")
        self.filesHaze = os.listdir(r"D:\OTS_original")

        self.filesClear_prefixes = self.build_prefix_table()

        # self.transform = transforms.Compose([
        #     transforms.RandomCrop([256,256]),
        #     transforms.ToTensor(),
        # ])

    def build_prefix_table(self):
        prefix_table = defaultdict(list)

        for file2 in self.filesClear:
            prefix = file2[:4]
            prefix_table[prefix].append(file2)
        return prefix_table

    def __getitem__(self, idx):

        idx = random.randint(0, 298234)#-35000)
        source_img = self.filesHaze[idx]
        prefix = source_img[:4]
        target_img = self.filesClear_prefixes[prefix]

        source_img = Image.open(r"D:\OTS_original\\" + source_img).convert("RGB")
        target_img = Image.open(r"D:\clear_images\\" + target_img[0]).convert("RGB")

        # crop_params = RandomCrop.get_params(source_img, [256, 256])
        # source_img = crop(source_img, *crop_params)
        # target_img = crop(target_img, *crop_params)

        source_img = ToTensor()(source_img)
        target_img = ToTensor()(target_img)





        return {'hazeimage': source_img, 'clearimage': target_img}



    def __len__(self):
        return min(len(self.filesClear), len(self.filesHaze))*100

class PairLoaderVal(Dataset):
    def __init__(self,):
        self.filesClear = os.listdir(r"D:\clear_images")
        self.filesHaze = os.listdir(r"D:\OTS_original")

        self.filesClear_prefixes = self.build_prefix_table()

        # self.transform = transforms.Compose([
        #     transforms.RandomCrop([256,256]),
        #     transforms.ToTensor(),
        # ])

    def build_prefix_table(self):
        prefix_table = defaultdict(list)

        for file2 in self.filesClear:
            prefix = file2[:4]
            prefix_table[prefix].append(file2)
        return prefix_table

    def __getitem__(self, idx):

        idx = random.randint(298234-35000,298234)
        source_img = self.filesHaze[idx]
        prefix = source_img[:4]
        target_img = self.filesClear_prefixes[prefix]

        source_img = Image.open(r"D:\OTS_original\\" + source_img).convert("RGB")
        target_img = Image.open(r"D:\clear_images\\" + target_img[0]).convert("RGB")

        crop_params = RandomCrop.get_params(source_img, [256, 256])
        source_img = crop(source_img, *crop_params)
        target_img = crop(target_img, *crop_params)

        source_img = ToTensor()(source_img)
        target_img = ToTensor()(target_img)





        return {'hazeimage': source_img, 'clearimage': target_img}



    def __len__(self):
        return min(len(self.filesClear), len(self.filesHaze))*1000

class TestDataset(Dataset):
    def __init__(self, ):
        self.filesClear = os.listdir(r"F:\pythonDoc\py\GT_480_640")
        self.filesHaze = os.listdir(r"F:\pythonDoc\py\Hazy_480_640")

        self.filesClear_prefixes = self.build_prefix_table()

        # self.transform = transforms.Compose([
        #     transforms.RandomCrop([256,256]),
        #     transforms.ToTensor(),
        # ])

    def build_prefix_table(self):
        prefix_table = defaultdict(list)

        for file2 in self.filesClear:
            prefix = file2[:4]
            prefix_table[prefix].append(file2)
        return prefix_table

    def __getitem__(self, idx):
        fileName = self.filesHaze[idx]

        source_img = Image.open(r"F:\pythonDoc\py\Hazy_480_640\\" + fileName).convert("RGB")
        target_img = Image.open(r"F:\pythonDoc\py\GT_480_640\\" + fileName).convert("RGB")



        source_img = ToTensor()(source_img)
        target_img = ToTensor()(target_img)

        return {'hazeimage': source_img, 'clearimage': target_img,"fileName":fileName}

        # return {'hazeimage': source_img, 'clearimage': target_img}

    def __len__(self):
        return min(len(self.filesClear), len(self.filesHaze))

class CNN2_(nn.Module):
    def __init__(self,M,N):
        super(CNN2_, self).__init__()
        self.inputChannel = 3
        self.M = M
        self.N = N
        self.tensor = []
        self.tensor2 = []

        m = 1
        n = 1
        for o in range(self.M):
            for p in range(self.N):
                conv_kernel = torch.zeros((self.inputChannel, self.inputChannel, self.M, self.N))
                for i in range(self.inputChannel):
                    for j in range(self.inputChannel):
                        for k in range(self.M):
                            for l in range(self.N):
                                if i==j and k==m-1 and l==n-1:
                                   conv_kernel[i,j,k,l] = 1
                self.tensor.append(conv_kernel)
                # indices = torch.nonzero(conv_kernel == 1)

                n = n+1
            m = m+1
            n=1

    def forward(self,x,i):


        out11 = F.conv2d(x,nn.Parameter(self.tensor[i],requires_grad=False),nn.Parameter(torch.zeros(3), requires_grad=False),stride=[self.M,self.N],padding=0)


        return out11




class PairLoader_(Dataset):
    def __init__(self,):
        self.files1 = os.listdir(r"D:\clear_images")
        self.files2 = os.listdir(r"D:\OTS_original")

        self.file1_prefixes = self.build_prefix_table()

        self.transform = transforms.Compose([
            transforms.Pad([0,0,129,33]),
            transforms.ToTensor(),
        ])
        self.cropNet = CNN2_(2,3)
        self.mask = torch.zeros([1, 256, 256])
        self.mask[:, :240, :213] = 1
    def build_prefix_table(self):
        prefix_table = defaultdict(list)

        for file2 in self.files1:
            prefix = file2[:4]
            prefix_table[prefix].append(file2)
        return prefix_table

    def __getitem__(self, idx):

        idx = random.randint(0, 298234)
        source_img = self.files2[idx]
        prefix = source_img[:4]
        target_img = self.file1_prefixes[prefix]

        source_img = Image.open(r"D:\OTS_original\\" + source_img).convert("RGB")
        target_img = Image.open(r"D:\clear_images\\" + target_img[0]).convert("RGB")


        source_img = self.transform(source_img).detach()
        target_img = self.transform(target_img).detach()



        target_img = target_img[:,:-1,:-1]


        originalImage = source_img[:,:-1,:-1]

        i = random.randint(0, 5)
        target_img = self.cropNet(target_img, i)
        originalImage = self.cropNet(originalImage,i)






        return {'hazeimage': originalImage, 'clearimage': target_img,"mask":self.mask}


        #return {'hazeimage': source_img, 'clearimage': target_img}

    def __len__(self):
        return min(len(self.files1), len(self.files2))*200
if __name__ == "__main__":
    data = PairLoader()
    i = 0
    while True:
        i = i+1
        a = data.__getitem__(i)

        hazeimage = (a['hazeimage'].cpu().numpy() * 255).astype(np.uint8)
        print(hazeimage.shape)
        hazeimage_ = np.transpose(hazeimage, (1, 2, 0))
        hazeimage_ = cv2.cvtColor(hazeimage_, cv2.COLOR_BGR2RGB)
        cv2.imshow('hazeimage', hazeimage_)


        clearimage = (a['clearimage'].cpu().numpy() * 255).astype(np.uint8)
        clearimage_ = np.transpose(clearimage, (1, 2, 0))
        clearimage_ = cv2.cvtColor(clearimage_, cv2.COLOR_BGR2RGB)
        cv2.imshow('clearimage', clearimage_)

        cv2.waitKey(0)

