import torch
import glob
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import os
import pickle
import numpy as np
import pdb
from PIL import Image

def getFileNamesFromFolder(input_path, tag):

    files = []
    for path in glob.glob(input_path + "*{}".format(tag)):
        files.append(os.path.basename(path))

    files = sorted(files)
    return files

class TrajDataset(data.Dataset):

    def __init__(self, image_sz, patch_sz, data_path, transform=None,is_train=1):     
        self.root_img = data_path
        self.root_scan = data_path
        self.is_train = is_train
        self.transform = transform
        self.image_sz = image_sz
        self.patch_sz = patch_sz

        self.feat_x = self.image_sz[0]/self.patch_sz[0]
        self.feat_y = self.image_sz[1]/self.patch_sz[1]
        self.total_patches = self.feat_x * self.feat_y
        self.BOS = self.total_patches+1
        self.EOS = self.total_patches+2
        self.vocab_sz = self.total_patches+3


        files_scanpaths = getFileNamesFromFolder(self.root_scan, ".mat")

        self.list_sample = sorted(files_scanpaths)


    def __getitem__(self, index):
        scan_basename = self.list_sample[index]
        #print(scan_basename)
        
        path_scans = os.path.join(self.root_scan, scan_basename)
        path_labels = os.path.join(self.root_scan+'/train_label/', scan_basename[:-4]+'_label.mat')
        path_img = os.path.join(self.root_img, scan_basename[:-8]+'.jpg')

        assert os.path.exists(path_scans), '[{}] does not exist'.format(path_scans)
        assert os.path.exists(path_img), '[{}] does not exist'.format(path_img)

        image = Image.open(path_img)
        org_image_sz = image.size[0]
        image = image.resize(self.image_sz, Image.LANCZOS)
        if self.transform is not None:
            image = self.transform(image)
        
        traj = sio.loadmat(path_scans)['traj']
        scanpaths = self.FixationtoclassID(traj)
        scanpaths = np.squeeze(scanpaths.astype(int))
        # print(scanpaths)
        labels = sio.loadmat(path_labels)['Y_label']
        labels = np.squeeze(labels.astype(int))

        labels = torch.from_numpy(labels)
        target = torch.from_numpy(scanpaths)

        return image, target, labels
        # return sample

    def __len__(self):
        return len(self.list_sample)


    def FixationtoclassID(self, traj):

        traj = traj[:,traj.min(axis=0)>=0]


        regionID_seq = np.zeros(len(traj[0])+2)
        feat_wtseq = np.zeros(int(self.feat_x*self.feat_y))
        regionID_seq [0] = self.feat_x*self.feat_y+1
        for pt in range(len(traj[0])):
            
            px = traj[0,pt]
            py = traj[1,pt]
            
            m = np.ceil((px*self.image_sz[0])/(self.patch_sz[0]*512))
            n = np.ceil((py*self.image_sz[1])/(self.patch_sz[1]*512))
            region_id = m+(n-1)*self.feat_x
            regionID_seq[pt+1] = region_id 

        end_token = self.feat_x*self.feat_y+2
        regionID_seq[len(traj[0])+1] = end_token
        regionID_seq = regionID_seq[regionID_seq<(self.feat_x*self.feat_y+3)]
        return regionID_seq


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, scanpath).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging scanpaths is not supported in default.

    Args:
        data: list of tuple (image, scanpaths). 
            - image: torch tensor .
            - scanpath: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, *, *).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded scanpath.
    """
    # Sort a data list by scanpath length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, scanpaths, labels = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    # labels = torch.stack(labels, 0)

    # Merge scanpaths (from tuple of 1D tensor to 2D tensor).
    lengths = [len(scan) for scan in scanpaths]
    targets = torch.zeros(len(scanpaths), max(lengths)).long()
    for i, scan in enumerate(scanpaths):
        end = lengths[i]
        targets[i, :end] = scan[:end]        
    return images, targets, lengths, labels

def get_loader(data_path, transform, image_sz, patch_sz, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom dataset."""
    # scanpath dataset
    traj_data = TrajDataset(image_sz, patch_sz, data_path=data_path,transform=transform)
    
    # Data loader for Scanpath dataset
    # This will return (images, scanpaths, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, *, *).
    # scanpaths: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each scanpaths. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=traj_data, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn) # , drop_last=True
    return data_loader