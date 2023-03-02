import torch
import pickle

from torch.utils.data import Dataset

from itertools import chain
from typing import Tuple

class MiniImagenetResized(Dataset):
    """ This is the resized version of the mini-imagenet dataset (84 x 84) - we also shuffle the sets this enables us to 'normal' instead of few-shot learning - can be downloaded from https://www.kaggle.com/datasets/whitemoon/miniimagenet """

    def __init__(self) -> None:
        """ """
        super(MiniImagenetResized, self).__init__()

        # Extract data from pickled files
        trn_imgs, trn_lbls = self.unpickle("data/mini-imagenet-cache-train.pkl")
        val_imgs, val_lbls = self.unpickle("data/mini-imagenet-cache-val.pkl")
        tst_imgs, tst_lbls = self.unpickle("data/mini-imagenet-cache-test.pkl")

        # Combine loaded data into a single imgs and lbls variables
        self.imgs = torch.cat((trn_imgs, val_imgs, tst_imgs), dim=0)
        del trn_imgs, val_imgs, tst_imgs
        self.lbls = dict(chain.from_iterable(d.items() for d in (trn_lbls, val_lbls, tst_lbls)))
        del trn_lbls, val_lbls, tst_lbls

        # Convert the labels to a tensor
        self.lbls = self.dict2labels()

    @staticmethod
    def unpickle(path) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Extract the images and classes from the specified pickle file """
        with open(path, "rb") as f:
            data = pickle.load(f)
            imgs = torch.swapaxes(torch.from_numpy(data["image_data"]), 1, 3)
            lbls = data["class_dict"]
        
        return (imgs, lbls)

    def dict2labels(self) -> torch.Tensor:
        """ Convert a dictionary to a label list """
        new_lbls = torch.zeros(size=(self.imgs.shape[0], len(self.lbls.keys())), dtype=torch.uint8)
        for class_idx, (_, sample_list) in enumerate(self.lbls.items()):
            new_lbls[sample_list, class_idx] = 1
        
        return new_lbls

    def __len__(self) -> int:
        """ Return the number of samples in the dataset """
        return self.imgs.shape[0]

    def __getitem__(self, index: int):
        """ Return a single sample from the dataset, both the images and labels """
        return self.imgs[index], self.lbls[index]