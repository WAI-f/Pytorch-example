from torchvision import datasets, transforms
from base import BaseDataLoader
from dataset.my_dataset import MyDataset
import os


class UNetDataLoader(BaseDataLoader):
    """
    Unet data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        self.dataset = MyDataset(os.path.join(self.data_dir, 'train_list.txt'))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
