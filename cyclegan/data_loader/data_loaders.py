import os
import glob
import random
import subprocess

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from cyclegan.base import BaseDataLoader


class BerkelyDataLoader(BaseDataLoader):

    options = [
        'apple2orange',
        'summer2winter_yosemite',
        'horse2zebra',
        'monet2photo',
        'cezanne2photo',
        'ukiyoe2photo',
        'vangogh2photo',
        'maps',
        'cityscapes',
        'facades',
        'iphone2dslr_flower',
        'ae_photo'
    ]

    def __init__(self, data_dir, dataset, batch_size, img_size, shuffle, validation_split,
                 num_workers, training=True):
        transforms_ = [
            transforms.Resize(int(img_size * 1.12), Image.BICUBIC),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        data_f = os.path.join(data_dir, dataset)
        if not os.path.exists(data_f):
            self.download_dataset(dataset)
        self.dataset = ImageDataset(data_f, transforms_=transforms_, unaligned=True)

        super(BerkelyDataLoader, self).__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers)

    def download_dataset(self, dataset):
        if dataset not in self.options:
            raise Exception(f'{dataset} is not a valid dataset. Options are {self.options}')
        subprocess.call(f'./download_dataset {dataset}', shell=True)


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
