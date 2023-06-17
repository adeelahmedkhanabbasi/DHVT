# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added LMDB dataset -- Youwei Liang
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import torch
import fnmatch
import numpy as np
import pandas as pd
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torch.utils.data import Dataset
from torchvision.io import read_image
from folder2lmdb import ImageFolderLMDB

class BtDataset(Dataset):
    def __init__(self, data_arr, img_dir, transform=None, target_transform=None):
        self.img_info = data_arr
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.dirs=os.listdir(img_dir)
        self.dirs.sort()

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,self.dirs[int(self.img_info.iloc[idx, 1])], self.img_info.iloc[idx, 0])
        image = read_image(img_path)
        label = torch.tensor(int(self.img_info.iloc[idx, 1]))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
def loadr():
    
    data_sets=["train","val"]
    train_list=[['id','label']]
    val_list=[['id','label']]
    for ds in data_sets:
      for root, dirs, files in os.walk("mydata/"+ds, topdown=False):
        for name in files:
          if name.endswith(('.jpg', '.jpeg', '.gif', '.png')):
              img_path=os.path.join(root, name)
            
              j=img_path
              pattern = 'Normal'
              if fnmatch.filter((j[i:i+len(pattern)] for i in range(len(j) - len(pattern))), pattern):
                a = np.array([[name,'0']]) 
              pattern = 'Sick'
              if fnmatch.filter((j[i:i+len(pattern)] for i in range(len(j) - len(pattern))), pattern):
                a = np.array([[name,'1']])
              if ds=="train":
                train_list=np.append(train_list,a, axis=0)
              else:
                val_list=np.append(val_list,a, axis=0)
    
    train_pd=pd.DataFrame(train_list[1:None],columns = ['id','label'])
    val_pd=pd.DataFrame(val_list[1:None],columns = ['id','label'])
    
    
    val_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                        ])
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize((224,224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(degrees=0.02),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                         ])
    training_data = BtDataset(data_arr=train_pd,img_dir='mydata/train',transform=train_transform)
    val_data = BtDataset(data_arr=val_pd,img_dir='mydata/val',transform=val_transform)
    #train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
    #val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True)
    
    return training_data, val_data


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'CIFAR-10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
        
    elif args.data_set in ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]:
        prefix = 'train' if is_train else 'val'
        root = os.path.join(args.data_path, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 345
        
    elif args.data_set == 'IMNET':
        if args.use_lmdb:
            root = os.path.join(args.data_path, 'train.lmdb' if is_train else 'val.lmdb')
            if not os.path.isfile(root):
                raise FileNotFoundError(f"LMDB dataset '{root}' is not found. "
                        "Pleaes first build it by running 'folder2lmdb.py'.")
            dataset = ImageFolderLMDB(root, transform=transform)
        else:
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
        
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'bt':
        nb_classes = 3
        img_mean, img_std = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)
        img_size = 72
        train_dataset,val_dataset= loadr()

    return train_dataset,val_dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
