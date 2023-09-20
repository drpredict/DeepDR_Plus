import os
import random
import cv2
from tqdm import tqdm

import numpy as np
from PIL import ImageFilter

import torch
import torchvision
import albumentations as aug
import albumentations.pytorch as aug_torch

from sklearn.linear_model import LogisticRegression

from . import pytorch_utils as ptu


# standard imagenet stats
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def accuracy_score(preds, labels):
    """ a simple top1 accuracy scoring function """
    if isinstance(preds, np.ndarray):
        return float((preds == labels).mean())
    else:
        return float((preds == labels).float().mean())


class MyCheckpoint(ptu.Checkpoint):
    """ an adaptation of ptu.Checkpoint for MoCo overriding batch_pass and agg_results"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def batch_pass(self,
                   device,
                   batch,
                   train,
                   *args, **kwargs):

        results = {}
        pbar_postfix = {}

        if self.model.moco:
            (q_img, k_img), labels = batch
            q_img = q_img.to(device)
            k_img = k_img.to(device)
            labels = labels.to(device)
            self.batch_size = q_img.shape[0]

            q, logits, zeros = self.model(q_img, k_img)

            loss = self.criterion(logits.float(), zeros.long())

            results['q'] = q.detach().cpu().numpy()
            results['labels'] = labels.detach().cpu().numpy()
        else:
            img, labels = batch
            img = img.to(device)
            labels = labels.to(device)
            self.batch_size = img.shape[0]

            out = self.model(img)

            loss = self.criterion(out.float(), labels.long())

            results['out'] = out.argmax(dim=1).detach().cpu().numpy()
            results['labels'] = labels.detach().cpu().numpy()
            pbar_postfix['score'] = self.score(
                results['labels'], results['out'])
            if len(self.raw_results) > 0:
                pbar_postfix['avg_score'] = self.score(np.concatenate(
                    self.raw_results['labels']), np.concatenate(self.raw_results['out']))

        return loss, results, pbar_postfix

    def agg_results(self, results, train):
        single_num_score = None
        additional_metrics = {}

        if self.model.moco:
            q = np.concatenate(results['q'])
            labels = np.concatenate(results['labels'])

            if train:
                self.model.clf = LogisticRegression(
                    **self.model.clf_hyperparams)
                self.model.clf.fit(q, labels)

            preds = self.model.clf.predict(q)
            single_num_score = self.score(labels, preds)
        else:
            preds = np.concatenate(results['out'])
            labels = np.concatenate(results['labels'])

            single_num_score = self.score(labels, preds)

        return single_num_score, additional_metrics


class Dataset(torch.utils.data.Dataset):
    """ a Dataset class for preloading data into memory """

    def __init__(self,
                 path: str,
                 transforms: torchvision.transforms.Compose,
                 ):
        """
        path : str
        """
        super().__init__()
        self.path = path
        self.transforms = transforms
        self.files = []
        for dirpath, _, fnames in os.walk(path):
            for fname in fnames:
                if fname.endswith('.jpg'):
                    self.files.append(os.path.join(dirpath, fname))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.imread(self.files[i])
        img = self.transforms(img)
        return img, 0


class Config:
    """ a simple class for managing experiment setup """

    def __call__(self):
        return vars(self)

    def __repr__(self):
        return str(self())

    def __str__(self):
        return self.__repr__()


class TwoCropsTransform:
    """ twice applied transforms to an image """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return self.transforms(image=x)['image'], self.transforms(image=x)['image']

    def __repr__(self):
        return str(self.transforms)

    def __str__(self):
        return self.__repr__()


class GaussianBlur:
    """ apply ImageFilter.GaussianBlur to an image """

    def __init__(self, sigma1=0.1, sigma2=2.0):
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __call__(self, x):
        return x.filter(ImageFilter.GaussianBlur(random.uniform(self.sigma1, self.sigma2)))

    def __repr__(self):
        return f'GaussianBlur({self.sigma1}, {self.sigma2})'

    def __str__(self):
        return self.__repr__()


moco_v1_transforms = TwoCropsTransform(aug.Compose([
    aug.SmallestMaxSize(
        max_size=512, always_apply=True),
    aug.CenterCrop(512, 512,
                   always_apply=True),
    aug.Flip(p=0.5),
    aug.ImageCompression(quality_lower=10, quality_upper=80, p=0.2),
    aug.MedianBlur(p=0.3),
    aug.RandomBrightnessContrast(p=0.5),
    aug.RandomGamma(p=0.2),
    aug.GaussNoise(p=0.2),
    aug.Rotate(border_mode=cv2.BORDER_CONSTANT,
               value=0, p=0.7, limit=45),
    aug.ToFloat(always_apply=True),
    aug_torch.ToTensorV2(),
])
)


moco_v2_transforms = TwoCropsTransform(aug.Compose([
    aug.SmallestMaxSize(
        max_size=512, always_apply=True),
    aug.CenterCrop(512, 512,
                   always_apply=True),
    aug.Flip(p=0.5),
    aug.ImageCompression(quality_lower=10, quality_upper=80, p=0.2),
    aug.MedianBlur(p=0.3),
    aug.RandomBrightnessContrast(p=0.5),
    aug.RandomGamma(p=0.2),
    aug.GaussNoise(p=0.2),
    aug.Rotate(border_mode=cv2.BORDER_CONSTANT,
               value=0, p=0.7, limit=45),
    aug.ToFloat(always_apply=True),
    aug_torch.ToTensorV2(),
]))


clf_train_transforms = aug.Compose([
    aug.SmallestMaxSize(
        max_size=512, always_apply=True),
    aug.CenterCrop(512, 512,
                   always_apply=True),
    aug.Flip(p=0.5),
    aug.ImageCompression(quality_lower=10, quality_upper=80, p=0.2),
    aug.MedianBlur(p=0.3),
    aug.RandomBrightnessContrast(p=0.5),
    aug.RandomGamma(p=0.2),
    aug.GaussNoise(p=0.2),
    aug.Rotate(border_mode=cv2.BORDER_CONSTANT,
               value=0, p=0.7, limit=45),
    aug.ToFloat(always_apply=True),
    aug_torch.ToTensorV2(),
])


clf_val_transforms = aug.Compose([
    aug.SmallestMaxSize(
        max_size=512, always_apply=True),
    aug.CenterCrop(512, 512,
                   always_apply=True),
    aug.ToFloat(always_apply=True),
    aug_torch.ToTensorV2(),
])
