import os
import glob
import random

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from cyclegan.base import BaseDataLoader


class Apples2OrangesDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, img_size, shuffle, validation_split,
                 num_workers, training=True):
        transforms_ = [
            transforms.Resize(int(img_size * 1.12), Image.BICUBIC),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.dataset = ImageDataset(data_dir, transforms_=transforms_, unaligned=True)

        super(Apples2OrangesDataLoader, self).__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers)


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
