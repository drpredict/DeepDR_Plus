
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision

from tqdm import tqdm

from src import arch
from src import utils
from config import cfg
from src import pytorch_utils as ptu

import warnings
warnings.filterwarnings("ignore")

print(cfg())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


print(cfg.moco.version)


model = arch.MoCo_v2(backbone=cfg.moco.backbone,
                     dim=cfg.moco.dim,
                     queue_size=cfg.moco.queue_size,
                     batch_size=cfg.moco.bs,
                     momentum=cfg.moco.model_momentum,
                     temperature=cfg.moco.temperature,
                     bias=cfg.moco.bias,
                     moco=True,
                     clf_hyperparams=cfg.moco.clf_kwargs,
                     seed=cfg.seed,
                     mlp=cfg.moco.mlp,
                     )

optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.moco.lr,
                            momentum=cfg.moco.optimizer_momentum,
                            weight_decay=cfg.moco.wd)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=cfg.moco.epochs,
                                                          eta_min=cfg.moco.min_lr) if cfg.moco.cos else None

checkpoint = utils.MyCheckpoint(version=cfg.moco.version,
                                model=model,
                                optimizer=optimizer,
                                criterion=nn.CrossEntropyLoss().to(device),
                                score=utils.accuracy_score,
                                lr_scheduler=lr_scheduler,
                                models_dir=cfg.models_dir,
                                seed=cfg.seed,
                                best_policy=cfg.moco.best_policy,
                                save=cfg.save,
                                )
if cfg.save:
    with open(os.path.join(checkpoint.version_dir, 'config.txt'), 'w') as f:
        f.writelines(str(cfg))

ptu.params(checkpoint.model)


# In[7]:


train_dataset = utils.Dataset(os.path.join(
    cfg.data_path, 'train'), cfg.moco.train_transforms)
train_eval_dataset = utils.Dataset(os.path.join(
    cfg.data_path, 'train'), cfg.moco.train_eval_transforms)
val_dataset = utils.Dataset(os.path.join(
    cfg.data_path, 'val'), cfg.moco.val_eval_transforms)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=checkpoint.model.batch_size,
    num_workers=cfg.num_workers,
    drop_last=True, shuffle=True, pin_memory=True)

train_eval_loader = torch.utils.data.DataLoader(
    train_eval_dataset,
    batch_size=checkpoint.model.batch_size,
    num_workers=cfg.num_workers,
    drop_last=True, shuffle=True, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=checkpoint.model.batch_size,
    num_workers=cfg.num_workers,
    drop_last=True, shuffle=False, pin_memory=True)


# In[ ]:


checkpoint.train(train_loader=train_loader,
                 train_eval_loader=train_eval_loader,
                 val_loader=val_loader,
                 train_epochs=int(
                     max(0, cfg.moco.epochs - checkpoint.get_log())),
                 optimizer_params=cfg.moco.optimizer_params,
                 prints=cfg.prints,
                 epochs_save=cfg.epochs_save,
                 epochs_evaluate_train=cfg.epochs_evaluate_train,
                 epochs_evaluate_validation=cfg.epochs_evaluate_validation,
                 device=device,
                 tqdm_bar=cfg.tqdm_bar,
                 save=cfg.save,
                 save_log=cfg.save_log,
                 )
