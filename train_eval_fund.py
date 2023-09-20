import math
from model import ModelProgression
from torch import nn
import torch
import numpy as np
from functools import cached_property
from trainer import Trainer
from torch.utils.data import Dataset
import pandas as pd
import cv2
import albumentations as aug
import albumentations.pytorch as aug_torch


class DeepSurModel(nn.Module):
    def __init__(self, K=512) -> None:
        super().__init__()
        self.K = K
        # sample parameters for the mixture model
        rnd = np.random.RandomState(12345)
        b = torch.FloatTensor(abs(rnd.normal(0, 10, (1, 1, self.K))+5.0))
        k = torch.FloatTensor(abs(rnd.normal(0, 10, (1, 1, self.K))+5.0))
        self.register_buffer('b', b)
        self.register_buffer('k', k)

        self.cnn = ModelProgression(backbone='resnet50', output_size=512)

    def _cdf_at(self, t):
        # pdf: nBatch * n * K
        pdf = 1 - torch.exp(-(1/self.b * (t)) ** self.k)
        return pdf

    def _pdf_at(self, t):
        # pdf: nBatch * n * K
        pdf = self._cdf_at(t)
        pdf = (1-pdf) * self.k * (1/self.b)*(t/self.b)**(self.k-1)
        return pdf

    def calculate_cdf(self, w, t):
        """
        Calculates the cumulative probability distribution function (CDF)
        for the given data.

        param w: nBatch * K: weights for mixture model
        param t: nBatch * n: target time to calculate pdf at
        return: nBatch * n: pdf values
        """
        t = t.unsqueeze(dim=2)
        w = nn.functional.softmax(w, dim=1)
        w = w.unsqueeze(dim=1)
        pdf = self._cdf_at(t)
        pdf = pdf * w
        pdf = pdf.sum(dim=2)
        return pdf

    def calculate_pdf(self, w, t):
        """
        Calculates the probability distribution function (pdf) for the given 
        data.

        param w: nBatch * K: weights for mixture model
        param t: nBatch * n: target time to calculate pdf at
        return: nBatch * n: pdf values
        """
        t = t.unsqueeze(dim=2)
        w = nn.functional.softmax(w, dim=1)
        w = w.unsqueeze(dim=1)
        pdf = self._pdf_at(t)
        pdf = pdf * w
        pdf = pdf.sum(dim=2)
        return pdf

    def calculate_survial_time(self, w, t_max=10, resolution=20):
        """
        Calculates the survival time for the given data.
        """
        t = torch.linspace(
            1/resolution,
            t_max,
            math.ceil(resolution*t_max)-1,
            dtype=torch.float32,
            device=w.device).view(1, -1)
        pdf = self.calculate_pdf(w, t)
        # print(pdf[])
        est = t.view(-1)[torch.argmax(pdf, dim=1)]
        # print(torch.argmax(pdf, dim=1).shape())
        return est

    def forward(self, x, t=None):
        x = self.cnn(x)
        if t is None:
            return x
        return x, self.calculate_cdf(x, t)


class ProgressionData(Dataset):

    def __init__(self, datasheet, transform):
        super().__init__()
        self.df = pd.read_csv(datasheet)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_file = self.df.iloc[idx]['image']
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        image = self.transform(image=image)['image']
        return dict(
            image=image,
            t1=self.df.iloc[idx]['t1'],
            t2=self.df.iloc[idx]['t2'],
            e=self.df.iloc[idx]['e'],
            # simulation only
            gt=self.df.iloc[idx]['gt'] if 'gt' in self.df.columns else 0,
        )


class TrainerDR(Trainer):

    @cached_property
    def model(self):
        model = DeepSurModel().to(self.device)
        if self.cfg.load_pretrain is not None:
            print('loading ', self.cfg.load_pretrain)
            print(model.cnn.backbone.load_state_dict(
                torch.load(self.cfg.load_pretrain, map_location=self.device)
            ))
        return model

    @cached_property
    def beta(self):
        return 1

    @cached_property
    def train_dataset(self):
        transform = aug.Compose([
            aug.SmallestMaxSize(
                max_size=self.cfg.image_size, always_apply=True),
            aug.CenterCrop(self.cfg.image_size, self.cfg.image_size,
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
        return ProgressionData('data_fund/train.csv', transform)

    @cached_property
    def test_dataset(self):
        transform = aug.Compose([
            aug.SmallestMaxSize(
                max_size=self.cfg.image_size, always_apply=True),
            aug.CenterCrop(self.cfg.image_size, self.cfg.image_size,
                           always_apply=True),
            aug.ToFloat(always_apply=True),
            aug_torch.ToTensorV2(),
        ])
        return ProgressionData('data_fund/test.csv', transform)

    @cached_property
    def optimizer(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-5)
        return optimizer

    def batch(self, epoch, i_batch, data) -> dict:
        # get and prepare data elements
        imgs = data['image'].to(self.device)
        t1 = data['t1'].to(self.device)
        t2 = data['t2'].to(self.device)
        e = data['e'].to(self.device)

        w, P = self.model(imgs, torch.stack([t1, t2], dim=1))
        P1 = P[:, 0]
        P2 = P[:, 1]
        loss = -torch.log(1-P1 + 0.000001) - torch.log(P2 +
                                                       0.000001) * self.beta * (e)
        loss += torch.abs(w).mean() * 0.00000001
        time_to_cal = torch.linspace(0, 20, 240).to(
            self.cfg.device).view(1, -1)
        cdf = self.model.calculate_cdf(w, time_to_cal)
        pdf = self.model.calculate_pdf(w, time_to_cal)
        survival_time = self.model.calculate_survial_time(w)
        return dict(
            loss=loss.mean(),
            pdf=pdf,
            cdf=cdf,
            t1=t1,
            t2=t2,
            survival_time=survival_time,
            gt=data['gt'],
        )

    def matrix(self, epoch, data) -> dict:
        return dict(
            loss=float(data['loss'].mean())
        )


if __name__ == '__main__':
    trainer = TrainerDR()
    trainer.train()
