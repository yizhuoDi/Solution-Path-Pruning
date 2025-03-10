# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import h5py
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
#from deit.folder2lmdb import ImageFolderLMDB
from torch.utils.data import Dataset
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from PIL import Image

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

class ImageNetHDF5(Dataset):
    def __init__(self, hdf5_path, dataset_type='train', transform=None):
        self.hdf5_file = h5py.File(hdf5_path, 'r')
        self.dataset_type = dataset_type
        self.transform = transform
        self.images = []
        self.labels = []

        # 循环遍历每个类别
        for class_folder in self.hdf5_file[f'{dataset_type}'].keys():
            self.images.extend(self.hdf5_file[f'{dataset_type}/{class_folder}/images'])
            self.labels.extend(self.hdf5_file[f'{dataset_type}/{class_folder}/labels'])

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def close(self):
        self.hdf5_file.close()


def build_dataset(is_train, args, client):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        # root = os.path.join(args.data_path, 'train' if is_train else 'val')
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform, client=client)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes

def build_dataset_lmdb(is_train, args, client):
    transform = build_transform(is_train, args)

        # traindir = os.path.join(args.data, 'train.lmdb')
        # valdir = os.path.join(args.data, 'val.lmdb')
        # train_dataset = ImageFolderLMDB(
        #     traindir,
        #     transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))
        # val_dataset = ImageFolderLMDB(
        #     valdir,
        #     transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))
    #if args.data_set == 'IMNET':
        # root = os.path.join(args.data_path, 'train' if is_train else 'val')
    root = os.path.join(args.data_path, 'train.lmdb' if is_train else 'val.lmdb')
    dataset = ImageFolderLMDB(root, transform=transform)
    nb_classes = 1000


    return dataset, nb_classes

def build_dataset_hdf5(is_train, args, client):
    transform = build_transform(is_train, args)
    
    #h5py.File(hdf5_file, 'r')
    
    root = os.path.join(args.data_path, 'imagenet.h5' )
    # fi=h5py.File(root, 'r')
    # print(fi['train'])
    # #print(fi['train'])
    # for item in fi.keys():
    #     print('main key is: {}'.format(item))
    #     content = fi[item][:]
    #     print(content.type())
    dataset = ImageNetHDF5(root,'train' if is_train else 'val' ,transform=transform)
    #dataset = ImageFolderLMDB(root, transform=transform)
    nb_classes = 1000
    

    return dataset, nb_classes


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
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
