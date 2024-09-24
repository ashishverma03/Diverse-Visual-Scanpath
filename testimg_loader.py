import torch
import glob
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import os
import pickle
import numpy as np
import nltk
import pdb
from PIL import Image

def getFileNamesFromFolder(input_path, tag):

    files = []
    for path in glob.glob(input_path + "*{}".format(tag)):
        files.append(os.path.basename(path))

    files = sorted(files)
    return files

class ImageDataset(data.Dataset):

    def __init__(self, image_sz, patch_sz, data_path, transform,is_train=1):     
        self.root_img = data_path
        self.transform = transform
        self.image_sz = image_sz
        self.patch_sz = patch_sz

        files_images = getFileNamesFromFolder(self.root_img, [".png", ".jpg", ".jpeg"])
        self.list_sample = sorted(files_images)

    def __getitem__(self, index):
        image_basename = self.list_sample[index]
        print(image_basename)

        path_img = os.path.join(self.root_img, image_basename)

        assert os.path.exists(path_img), '[{}] does not exist'.format(path_img)

        image = Image.open(path_img)
        img_org_sz = image.size
        image = image.resize(self.image_sz, Image.LANCZOS)
        if self.transform is not None:
            image = self.transform(image)

        return image, image_basename, img_org_sz

    def __len__(self):
        return len(self.list_sample)

def get_loader(data_path, transform, image_sz, patch_sz, shuffle):
    """Returns torch.utils.data.DataLoader for custom dataset."""
    test_images = ImageDataset(image_sz, patch_sz, data_path=data_path,transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=test_images,
                                              shuffle=shuffle)
    return data_loader
