import os
import time
from typing import Union, Optional, Callable, Any, Dict, List, Tuple
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def naming_scheme(version: str,
                  epoch: Union[int, str],
                  epoch_fmt: Optional[str]="{:03}") -> str:
    """a  func for converting a comb of version, epoch to a filename str with a fixed naming_scheme

    Parameters
    ----------
    version : str (or str like)
        The version name of the Checkpoint
    epoch : str or int
        the save type: -1 for last model, an int for a specific epoch, 'best' for best epoch
    epoch_fmt : str
        str format for epoch (default is "{:03}")

    Returns
    -------
    str
        filename
    """
    if not isinstance(epoch, str):
        epoch = epoch_fmt.format(epoch)
    return 'checkpoint_{:}_epoch{:}'.format(version, epoch)


def load_model(version: str=None,
               models_dir: str=None,
               epoch: Union[int, str]=-1,
               naming_scheme: Optional[Callable[[str, Union[int, str], str], str]]=naming_scheme,
               log: bool=False,
               explicit_file: Optional[str]=None):
    """a func for loading a Checkpoint using a comb of version, epoch usind the dill module

    Parameters
    ----------
    version : convertable to str, optional if is given explicit_file
        The version name of the Checkpoint (default is None)
    models_dir : str, optional if is given explicit_file
        The full or relative path to the versions dir (default is None)
    epoch : str or int, optional
        the save type: '-1' for last model, an int for a specific epoch, 'best' for best epoch (default is -1)
    prints : bool, optional
        if prints=True some training statistics will be printed (default is True)
    naming_scheme : callable(version, epoch), optional
        a func that gets version, epoch and returns a str (default is naming_scheme)
    explicit_file : str, optional
        an explicit path to a Checkpoint file (default is None),
        if explicit_file is not None, ignores other args and loads explicit_file 

    Returns
    -------
    Checkpoint
        the loaded Checkpoint
    """
    if log:
        log_path = os.path.join(models_dir, str(version), naming_scheme(version, epoch)) + '_log.csv'
        train_batch_log_path = os.path.join(models_dir, str(version), naming_scheme(version, epoch)) + '_train_loss_log.csv'
        log = pd.read_csv(log_path, index_col='Unnamed: 0')
        train_batch_log = pd.read_csv(train_batch_log_path, index_col='Unnamed: 0')
        return log, train_batch_log
    else:
        import dill

        if explicit_file is None:
            model_path = os.path.join(models_dir, str(version), naming_scheme(version, epoch) + '.pth')
        else:
            if version is not None or models_dir is not None:
                warnings.warn(f'\n\nexplicit_file={explicit_file} was specified\nignoring version={version}, models_dir={models_dir}\n')
            model_path = explicit_file
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), pickle_module=dill)

        return checkpoint


def set_p_dropout(model: nn.Module,
                  p: float,
                  max_rec_depth: int=50,
                  i: int=0):
    """
    set p of all nn.Dropout modules in model to p, recursively

    Parameters
    ----------
    model : nn.Module
        The model
    p : 0 < float < 1
        the new dropout p
    max_rec_depth : int
        the max recursively
    i : int
        current recursion depth
    """
    assert isinstance(p, float) and p > 0.0 and p < 1.0, 'p must be a float: 0.0 < p < 1.0'
    if isinstance(model, nn.Dropout):
        model.p = p
    elif i < max_rec_depth and isinstance(model, nn.Module):
        for module in model._modules.values():
            if module is not model:
                set_p_dropout(module, p, max_rec_depth, i+1)


def params(model: nn.Module) -> None:
    """ prints the total number of parameters and number of trainable parameters for a given model """
    print("Number of parameters {} ".format(sum(param.numel() for param in model.parameters())) + 
          "trainable {}".format(sum(param.numel() for param in model.parameters() if param.requires_grad)))


@contextmanager
def set_temp_seed(seed: int):
    """
    a context manager which temporarily sets a fixed random seed for torch.random,
    then returns random number generator back to the previous state.

    Parameters
    ----------
    seed : int
        temporary seed number
    """
    prev_state = torch.random.get_rng_state()
    try:
        torch.random.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(prev_state)

