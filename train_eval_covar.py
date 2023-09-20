from model import ModelProgression
from torch import nn
import torch
import numpy as np
from functools import cached_property
from trainer import Trainer
from torch.utils.data import Dataset
import pandas as pd
import torch.nn.functional as F
from train_eval_fund import DeepSurModel as _DeepSurModel


class DeepSurModel(_DeepSurModel):
    def __init__(self, K=512) -> None:
        super().__init__()
        self.K = K
        # sample parameters for the mixture model
        rnd = np.random.RandomState(12345)
        b = torch.FloatTensor(abs(rnd.normal(0, 10, (1, 1, self.K))+5.0))
        k = torch.FloatTensor(abs(rnd.normal(0, 10, (1, 1, self.K))+5.0))
        self.register_buffer('b', b)
        self.register_buffer('k', k)

        self.cnn = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.K),
        )

    def forward(self, x, t=None):
        x = self.cnn(x)
        if t is None:
            return x
        return x, self.calculate_cdf(x, t)


class ProgressionData(Dataset):

    def __init__(self, datasheet, feature_keys):
        super().__init__()
        self.df = pd.read_csv(datasheet)
        self.feature_keys = feature_keys

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = self.df.iloc[idx][self.feature_keys].values
        return dict(
            covar=X,
            t1=self.df.iloc[idx]['t1'],
            t2=self.df.iloc[idx]['t2'],
            e=self.df.iloc[idx]['e'],
            # simulation only
            gt=self.df.iloc[idx]['gt'] if 'gt' in self.df.columns else 0,
        )


class TrainerDR(Trainer):
    def __init__(self, feature_columns=None) -> None:
        super().__init__()
        self.feature_columns = feature_columns

    @cached_property
    def model(self):
        return DeepSurModel().to(self.device)

    @cached_property
    def beta(self):
        return 1

    @cached_property
    def train_dataset(self):
        return ProgressionData('data_covar/train.csv', self.feature_columns)

    @cached_property
    def test_dataset(self):
        return ProgressionData('data_covar/test.csv', self.feature_columns)

    @cached_property
    def optimizer(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=1e-5)
        return optimizer

    def batch(self, epoch, i_batch, data) -> dict:
        # get and prepare data elements
        co_var = data['covar'].to(self.device).to(torch.float32)
        if co_var.shape[1] < 32:
            co_var = F.pad(co_var, (0, 32-co_var.shape[1]))
        t1 = data['t1'].to(self.device)
        t2 = data['t2'].to(self.device)
        e = data['e'].to(self.device)

        w, P = self.model(co_var, torch.stack([t1, t2], dim=1))
        P1 = P[:, 0]
        P2 = P[:, 1]
        loss = -(
            torch.log(1-P1 + 0.000001) +
            torch.log(P2 + 0.000001) * self.beta * (e)
        )
        loss += torch.abs(w).mean() * 0.00000001
        time_to_cal = torch.linspace(0.1, 20, 240).to(
            self.cfg.device).view(1, -1)
        cdf = self.model.calculate_cdf(w, time_to_cal)
        pdf = self.model.calculate_pdf(w, time_to_cal)
        survival_time = self.model.calculate_survial_time(w)
        return dict(
            loss=loss.mean(),
            cdf=cdf,
            pdf=pdf,
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
    import sys
    trainer = TrainerDR(sys.argv[1:])
    trainer.train()
