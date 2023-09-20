import json
import os
import pickle
import time
from collections import defaultdict
from functools import cached_property
from typing import Dict

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

# from .survivalreg.label_coder import LabelCoder
# from .survivalreg.model import ModelProgression
# from .survivalreg.util.config import Config, Parser
from typing import Any
import os
import json


class Parser(object):
    def __init__(self,
                 name: str, default: Any = None,
                 type_: callable = None, help_info: str = None):
        self.name = name
        self.default = default
        self.type_ = type_
        self.help = help_info

    def __set__(self, instance, value):
        setattr(self, 'value', value)

    def __get__(self, instance, owner):
        if hasattr(self, 'value'):
            return getattr(self, 'value')
        v = os.environ.get(self.name, self.default)
        if self.type_ is not None and self.name in os.environ:
            v = self.type_(v)
        setattr(self, 'value', v)
        return v

    def __call__(self, s):
        if self.type_ is not None:
            return self.type_(s)
        return s

    def __str__(self):
        return f'<{self.name}: {self.help} default:{self.default}>'

    def __repr__(self) -> str:
        return self.__str__()


class Config(object):
    def __init__(self):
        pass

    @property
    def value_dict(self):
        return self._search_cfg_recursively(self.__class__)

    @staticmethod
    def _search_cfg_recursively(root):
        vals = dict()
        for base in root.__bases__:
            vals.update(Config._search_cfg_recursively(base))
        for k, v in root.__dict__.items():
            if isinstance(v, Parser):
                vals[k] = v.__get__(None, None)
        return vals

    def __repr__(self):
        return f'<{self.__class__.__name__}: {json.dumps(self.value_dict)}>'

    def sample_cfg(self):
        for k, v in self.__class__.__dict__.items():
            if isinstance(v, Parser):
                print(f'{k}={v.default} ', end='')
        print()


class TrainerConfig(Config):
    debug = Parser('debug', False,
                   lambda x: not x.lower().startswith('f'), 'debug mode ')
    load_pretrain = Parser('load_pretrain', None, str, 'load pretrained model')
    batch_size = Parser('batch_size', 32, int, 'batch size')
    epochs = Parser('epochs', 100, int, 'number of max epochs to train')
    image_size = Parser('image_size', 512, int, 'image size')
    lr = Parser('lr', 0.001, float, 'learning rate')
    device = Parser('device', 'cuda:0', str, 'device')
    num_workers = Parser('num_workers', 4, int, 'number of workers')
    model = Parser('model', 'resnet50', str, 'backbone model')


class Trainer():
    cfg = TrainerConfig()
    # label_coder: LabelCoder = None

    def __init__(self) -> None:
        print(self.__class__)
        tt = time.gmtime()
        self.running_uuid = f'{tt.tm_year}{tt.tm_mon:02d}{tt.tm_mday:02d}_{tt.tm_hour:02d}{tt.tm_min:02d}{tt.tm_sec:02d}'
        print('running_uuid', self.running_uuid)
        print(self.cfg)
        self.epoch = None
        self._result_cache: defaultdict = None
        self._pretrain_loaded = False

    def _get_cfg_recursive(self, cls=None):
        if cls is None:
            cls = self.__class__
        parent_cfg = Config()
        for parent in cls.__bases__:
            parent_cfg.__dict__.update(
                self._get_cfg_recursive(parent).__dict__)
        if hasattr(cls, '_cfg'):
            cfg = cls._cfg()
            parent_cfg.__dict__.update(cfg.__dict__)
        return parent_cfg

    @cached_property
    def logger_dir(self):
        logger_dir = f'logs/{self.__class__.__name__}_{self.running_uuid}'
        if self.cfg.debug:
            logger_dir = f'logs/debug_{self.running_uuid}'
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        last_link_dir = f'logs/{self.__class__.__name__}_last'
        if os.path.islink(last_link_dir):
            os.remove(last_link_dir)
        os.symlink(
            f'{self.__class__.__name__}_{self.running_uuid}', last_link_dir)
        return logger_dir

    @cached_property
    def training_log(self):
        training_log = open(os.path.join(
            self.logger_dir, f'training_log.txt'), 'w')
        return training_log

    @cached_property
    def model(self):
        raise NotImplementedError

    @cached_property
    def train_dataset(self) -> Dataset:
        raise NotImplementedError

    @cached_property
    def test_dataset(self) -> Dataset:
        raise NotImplementedError

    @cached_property
    def train_loader(self) -> DataLoader:
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )
        return loader

    @cached_property
    def test_loader(self) -> DataLoader:
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )
        return loader

    @cached_property
    def device(self):
        if self.cfg.device.startswith('cuda') and not torch.cuda.is_available():
            print('cuda is not available, using CPU mode')
            self.cfg.device = 'cpu'
        device = torch.device(self.cfg.device)
        return device

    @cached_property
    def optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        return optimizer

    @cached_property
    def criterion(self):
        raise NotImplementedError

    @cached_property
    def scheduler(self):
        sch = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.1)
        self.optimizer.zero_grad()
        self.optimizer.step()
        return sch

    def batch(self, epoch, i_batch, data) -> dict:
        x1, x2, l1, l2, dt = data
        code_1 = self.label_coder(l1)
        code_2 = self.label_coder(l2)
        x1 = x1.to(torch.float32).to(self.device)
        x2 = x2.to(torch.float32).to(self.device)
        code_1 = code_1.to(torch.float32).to(self.device)
        code_2 = code_2.to(torch.float32).to(self.device)
        dt = dt.to(torch.float32).to(self.device)
        y1 = self.model(x1)
        y2 = self.model(x2)
        loss = self.criterion(y1, y2, code_1, code_2, dt)
        return dict(
            loss=loss,
            y1=y1,
            y2=y2,
            code_1=code_1,
            code_2=code_2,
            dt=dt,
        )

    def matrix(self, epoch, data) -> dict:
        mean_loss = torch.mean(data['loss']).item()
        n_auc = data['code_1'].size(1)
        mat_dict = {'mean_loss': mean_loss}
        pred = torch.cat([data['y1'], data['y2']])
        label = torch.cat([data['code_1'], data['code_2']])
        for i in range(n_auc):
            try:
                mat_dict[f'auc_{i}'] = roc_auc_score(
                    label[:, i] > 0, pred[:, i])
            except Exception as e:
                print(e)
        return mat_dict

    def collect_result(self, output: Dict):
        if self._result_cache is None:
            self._result_cache = defaultdict(list)
        for k, v in output.items():
            self._result_cache[k].append(v.detach().cpu())

    def merge_result(self):
        collected = {}
        for k, v in self._result_cache.items():
            if len(v[0].shape) == 0:
                collected[k] = torch.stack(v)
            else:
                collected[k] = torch.cat(v)
        self._result_cache.clear()
        return collected

    def train(self):
        print(self.cfg, file=self.training_log)
        print(self.scheduler)
        print(self.optimizer)
        for epoch in range(self.cfg.epochs):
            self.epoch = epoch
            self.model.train()
            self.model.to(self.device)
            for i_batch, batch_data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.batch(epoch, i_batch, batch_data)
                loss = output['loss']
                loss.backward()
                self.optimizer.step()

                print(f'training {self.running_uuid} epoch:{epoch}/{self.cfg.epochs} '
                      f'batch {i_batch}/{len(self.train_loader)} {loss}', end='\r')
                print(json.dumps(dict(
                    type='train',
                    epoch=epoch,
                    ibatch=i_batch,
                    loss=float(loss),
                )), file=self.training_log)
                self.training_log.flush()
                self.collect_result(output)
                if self.cfg.debug and i_batch > 20:
                    break
            self.scheduler.step()
            torch.save(self.model.state_dict(), os.path.join(
                self.logger_dir, f'model_{epoch:03d}.pth'))
            # calculate matrix training
            metrix = self.matrix(epoch=self.epoch, data=self.merge_result())
            # print(metrix)
            metrix.update(dict(
                type='train matrix',
                epoch=self.epoch,
            ))
            print(json.dumps(metrix), file=self.training_log)
            self.training_log.flush()
            print('epoch train mat ', self.epoch, end=' ')
            for k, v in metrix.items():
                print(f'{k}: {v}', end=' ')
            print()

            self.test()
            if self.cfg.debug:
                break

    def predict(self, dataset):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for i_batch, data in enumerate(dataset):
                output = self.batch(epoch=self.epoch, i_batch=-1, data=data)
                self.collect_result(output)
                print(
                    f'predicting {self.running_uuid} epoch:{self.epoch}/'
                    f'{self.cfg.epochs} batch {i_batch}/{len(dataset)}',
                    end=' \r')
                if self.cfg.debug and i_batch > 2:
                    break
        merged_output = self.merge_result()
        return merged_output

    def test(self):
        merged_output = self.predict(dataset=self.test_loader)
        metrix = self.matrix(epoch=self.epoch, data=merged_output)
        metrix.update(dict(
            type='test matrix',
            epoch=self.epoch,
        ))
        print(json.dumps(metrix), file=self.training_log)
        self.training_log.flush()
        print('Test epoch', self.epoch, end=' ')
        for k, v in metrix.items():
            print(f'{k}: {v}', end=' ')
        print()
        pickle.dump(merged_output, open(os.path.join(
            self.logger_dir, f'preds_{self.epoch}.pkl'), 'wb'))
