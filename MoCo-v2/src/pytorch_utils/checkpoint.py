"""

"""


import os
import time
from typing import Union, Optional, Callable, Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from .utils import naming_scheme, set_temp_seed


class Checkpoint:
    """
    a wrapper class that manages all interactions with an nn.Module instance,
    including training, saving, evaluating, stats logging,
    setting training hyperparameters, plotting training stats, monitoring training

    Attributes
    -------
    version : convertable to str
        an identifier for the class instance version
    seed : int
        a constant seed to be used in all model interactions
    models_dir : str
        path to be used for versions dirs
    model : nn.Module instance
        the model
    optimizer : Object
        the optimizer
    criterion : callable
        the loss function
    naming_scheme : callable
        the naming_scheme of the class instance, see naming_scheme for an example
    score : callable
        the scoring function to be used for evaluating the train ans validation score,
        can be any function that gets y_true, y_pred args,
        Example : >>> sklearn.metrics.roc_auc_score(y_true, y_pred)
    decision_func : callable
        an optional feature for the case when there needs to be another transformation
        on the raw model output before passing it to the scoring function
        (default is lambda x : x)
    log : pd.DataFrame
        a pd.DataFrame that loggs stats each epoch
    """

    def __init__(self,
                 version: str,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.modules.loss._Loss,
                 score: Callable[[np.ndarray, np.ndarray], float],
                 lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 models_dir: str = 'models',
                 seed: int = None,
                 best_policy: str = 'val_loss',
                 naming_scheme: Optional[Callable[[
                     str, Union[int, str], int, str], str]] = naming_scheme,
                 save: bool = False,
                 ):
        """
        Parameters
        -------
        models_dir : str
            path to be used for versions dirs
        version : convertable to str
            an identifier for the class instance version
        model : nn.Module instance
            the model
        score : callable
            the scoring function to be used for evaluating the train ans validation score,
            can be any function that gets y_true, y_pred args,
            Example : >>> sklearn.metrics.roc_auc_score(y_true, y_pred)
        decision_func : callable, optional
            an optional feature for the case when there needs to be another transformation
            on the raw model output before passing it to the scoring function
            (default is lambda x : x)
        seed : int, optional
            a constant seed to be used in all model interactions (default is 42)
        optimizer : Object
            the initialized optimizer
        criterion : Object, optional
            the loss function (default is nn.BCELoss())
        naming_scheme : callable, optional
            the naming_scheme of the class instance (default is naming_scheme)
        save : bool, optional
            if save=True, saves the class instance (default is False)

        Examples
        --------
        >>> model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
        >>> checkpoint = Checkpoint(version=1.0.0,
        >>>                         model=model,
        >>>                         optimizer=torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=4e-4),
        >>>                         criterion=nn.BCELoss(),
        >>>                         score=sklearn.metrics.roc_auc_score,
        >>>                         models_dir='models',
        >>>                         seed=42,
        >>>                         naming_scheme=naming_scheme,
        >>>                         save=False)
        """
        self.version = version
        self.seed = seed
        self.models_dir = models_dir
        self.naming_scheme = naming_scheme
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.score = score
        self.lr_scheduler = lr_scheduler
        self.best_policy = best_policy

        self.temp_results = {}

        log_columns = ['train_time',
                       'timestamp',
                       'train_loss',
                       'val_loss',
                       'train_score',
                       'val_score',
                       'batch_size',
                       'best',
                       ]

        log_dtypes = {'train_time': np.datetime64,
                      'timestamp': np.float64,
                      'train_loss': np.float64,
                      'val_loss': np.float64,
                      'train_score': np.float64,
                      'val_score': np.float64,
                      'batch_size': np.int64,
                      'best': np.bool,
                      }

        self.log = pd.DataFrame(columns=log_columns).astype(dtype=log_dtypes)

        train_loss_log_columns = ['epoch',
                                  'batch',
                                  'timestamp',
                                  'loss',
                                  ]

        train_loss_log_dtypes = {'epoch': np.int64,
                                 'batch': np.int64,
                                 'timestamp': np.float64,
                                 'loss': np.float64,
                                 }

        self.train_loss_log = pd.DataFrame(
            columns=train_loss_log_columns).astype(dtype=train_loss_log_dtypes)

        if save:
            self.save()

    def run(self, *args, **kwargs):
        """ overrideable run function, for example see _run """
        return self._run(*args, **kwargs)

    def batch_pass(self, *args, **kwargs):
        """ overrideable batch_pass function, for example see _batch_pass """
        return self._batch_pass(*args, **kwargs)

    def agg_results(self, *args, **kwargs):
        """ overrideable agg_results function, for example see _agg_results """
        return self._agg_results(*args, **kwargs)

    def callback(self, *args, **kwargs) -> None:
        """ overrideable callback function """
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(self.get_log(
                    col=self.best_policy, epoch=-1), epoch=None)
            else:
                self.lr_scheduler.step()

    def __str__(self) -> str:
        return f"Checkpoint(version={version}, model={model}, optimizer={optimizer})"

    def __repr__(self) -> str:
        return str(self)

    def get_log(self,
                col: Optional[str] = 'epoch',
                epoch: Union[int, str] = -1) -> Any:
        """
        extracts stats from self.log

        Parameters
        -------
        col : str, optional
            path to be used for versions dirs (default is 'epoch')
        epoch : int or str, optional
            the epoch to load, can be int > 0 or -1 or 'best'  (default is -1)

        Returns
        -------
        str or int or float or None
            returns the log stat in the index (line) epoch
            (if epoch epoch == -1 returns last line,
            if epoch epoch == 'best' returns last index (line) where self.log['best'] == True)
            returns None if epoch index or col column does not exist in the log
        """
        if len(self.log) == 0:
            if col == 'val_loss' or col == 'train_loss':
                return np.inf
            else:
                return 0

        if epoch == 'best':
            try:
                index = self.log[self.log['best'] == True].iloc[-1].name
            except Exception as e:
                index = self.log.iloc[-1].name
        elif isinstance(int(epoch), int) and epoch > 0:
            index = int(self.log.loc[epoch].name)
        else:
            index = int(self.log.iloc[epoch].name)

        if col == 'epoch':
            return index
        else:
            try:
                return self.log[col].loc[index]
            except Exception:
                return None

    def _get_optimizer_params(self) -> dict:
        return dict(sorted(list({key: val for key, val in self.optimizer.param_groups[0].items() if key != 'params'}.items()), key=lambda x: x[0]))

    def save(self,
             epoch: int = 0,
             ) -> None:
        """ saves the class instance using self.naming_scheme

        Parameters
        -------
        best : bool, optional
            aditionally saves the model to a 'best' epoch file (default is False)
            Example file name : 'checkpoint_1_epoch-best.pth'
        epoch : bool, optional
            aditionally saves the model to an epoch file (default is False)
            Example file name : 'checkpoint_1_epoch-001.pth'
        log : str, optional
            if log=True, saves log and train_loss_log to csv in version_dir (default is False)
        explicit_file : str, optional
            if explicit_file is not None, saves the model to an explicitly specified explicit_file name (default is None)
        """
        self.version_dir = os.path.join(self.models_dir, self.version)
        if not os.path.exists(self.models_dir):
            os.mkdir(self.models_dir)
        if not os.path.exists(self.version_dir):
            os.mkdir(self.version_dir)

        if epoch:
            torch.save(self.model.q_encoder.state_dict(), os.path.join(
                self.version_dir, f'{epoch:03d}.pth'))

    def plot_checkpoint(self,
                        attributes: List[str],
                        plot_title: str,
                        y_label: str,
                        scale: str = 'linear',
                        base: int = 10,
                        save: bool = False) -> None:
        """ plots stats of the class instance
        Parameters
        ----------
        attributes : iterable of str
            an iterable of self.log.columns to plot
        plot_title : str
            the plot title to display
        y_label : str
            the y label to display
        scale : str, optional
            plot scale, if scale='log' plots y axis in log scale,
            otherwise plots y axis in linear scale (default is 'linear')
        base : int, optional
            used if scale='log', log base y axis (default is 10)
        save : bool, optional
            saves plot to <plot_title>.png in version dir (default is False)
        """
        if not self.get_log():
            print("model have not trained yet")
            return
        epochs = self.log.index
        to_plot = []
        try:
            for attribute in attributes:
                to_plot.append(self.log[attribute])
        except Exception as e:
            print('attributes must be an iterable')
            raise e
        min_e = int(np.min(epochs))
        max_e = int(np.max(epochs))
        for data in to_plot:
            plt.plot(epochs, data)
        plt.xlim(min_e - (max_e - min_e)*0.02, max_e + (max_e - min_e)*0.02)
        plt.xlabel('epoch')
        plt.ylabel(y_label)
        if scale == 'log':
            plt.yscale(scale, base=base)
        else:
            plt.yscale(scale)
        plt.legend(attributes)
        plt.title(plot_title)
        if save:
            plt.savefig(os.path.join(self.models_dir, self.version,
                        '{}.png'.format(plot_title)), dpi=200)
        plt.show()

    def _batch_pass(self,
                    device: torch.device,
                    batch: Tuple[Any],
                    train: bool,
                    *args, **kwargs) -> Tuple[torch.Tensor, dict]:
        """ overrideable through self.batch_pass
        a function for:
            1. calculating total batch loss for optimizer step
            2. calculating batch results needed for scoring (or other) metrics
            3. (optional) performing batch transformations outside of the model, for example: max, argmax, etc.

        Parameters
        ----------
        device : torch device or str
            the device for the pass
        batch : object
            a batch from the loader
        args : optional
            can be passed through *args, **kwargs to self.train
        kwargs : optional
            can be passed through *args, **kwargs to self.train

        Returns
        -------
        torch.Tensor
            torch loss for backprop (can be a sum of multiple lossed)
        dict
            additional results, the results.values will be combined to a list where each list entry reffers to a batch
        """
        X = [b.to(device) for b in batch[:-1]]
        y = batch[-1].to(device)

        self.batch_size = y.shape[0]

        out = self.model(*X)
        loss = self.criterion(out, y.float())

        results = {
            'preds': out.detach().cpu().numpy(),
            'trues': y.detach().cpu().numpy()
        }

        pbar_postfix = {
            'loss': float(loss.data)
        }

        return loss, results, pbar_postfix

    def _agg_results(self,
                     results: Dict[str, List[np.ndarray]],
                     train: bool) -> Tuple[float, dict]:
        """ overrideable through self.agg_results
        a function for:
            1. aggregating results of the epoch, from each _batch_pass (or batch_pass)
            2. returning a single_num_score
            3. returning additional stats which will be logged to self.log

        Parameters
        ----------
        results : dict
            a dict of
                key: keys from _batch_pass (or batch_pass)
                val: a list of values from _batch_pass (or batch_pass), each list entry reffers to a batch

        Returns
        -------
        single_num_score
            a single num score for the whole pass
        additional_metrics : dict, optional
            a dict of additional metrics to be logged in self.log
        """
        preds = np.concatenate(results['preds'])
        trues = np.concatenate(results['trues'])

        single_num_score = self.score(trues, preds)
        additional_metrics = {}

        return single_num_score, additional_metrics

    def _run(self,
             device: torch.device,
             loader: torch.utils.data.DataLoader,
             epoch: int,
             train: bool,
             tqdm_bar: bool = False,
             max_iterations: int = None,
             *args, **kwargs) -> Tuple[float, float, dict]:
        """
        a private method used to pass data through model
        if train=True : computes gradients and updates model weights
        if train=False : returns loss and score
        """
        # init evaluation results
        self.losses = []
        self.raw_results = {}

        if tqdm_bar:
            from tqdm import tqdm
            loader = tqdm(loader, total=len(loader))

        # mini-batch loop
        for i, batch in enumerate(loader):
            # pass batch through model, calculate loss and predict
            loss, batch_results, pbar_postfix = self.batch_pass(
                device, batch, train=train, *args, **kwargs)

            if tqdm_bar:
                if len(self.losses) > 0:
                    pbar_postfix.update(
                        {'loss': float(loss.data), 'avg_loss': sum(self.losses)/len(self.losses)})
                loader.set_postfix(pbar_postfix)

            assert isinstance(
                batch_results, dict), 'batch_results returned by batch_pass must be a dict'

            if self.train_mode:
                # update model weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                row = {
                    'epoch': epoch,
                    'batch': i+1,
                    'timestamp': time.strftime('%H:%M:%S %d-%m-%Y'),
                    'loss': float(loss.data),
                }
                row.update(pbar_postfix)

                self.train_loss_log = self.train_loss_log.append(pd.Series(row, name=1 if len(
                    self.train_loss_log.index) == 0 else (max(self.train_loss_log.index) + 1)), ignore_index=False)
            # update evaluation results
            self.losses.append(float(loss.data))
            for key, val in batch_results.items():
                if key not in self.raw_results:
                    self.raw_results[key] = list()
                self.raw_results[key].append(val)

            # max_iterations
            if max_iterations is not None and i+1 >= max_iterations:
                break

        # single_num_score, results = self.agg_results(self.raw_results, train=train)
        # assert isinstance(results, dict), 'results returned by agg_results must be a dict'
        return float(np.array(self.losses).mean()), {}, {}

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              train_eval_loader: torch.utils.data.DataLoader = None,
              val_loader: torch.utils.data.DataLoader = None,
              train_epochs: int = 100,
              optimizer_params: dict = {},
              prints: str = 'display',
              callback_kwargs: Optional[dict] = {},
              device: torch.device = torch.device(
                  "cuda" if torch.cuda.is_available() else "cpu"),
              save: bool = False,
              epochs_save: int = None,
              epochs_evaluate_train: int = 1,
              epochs_evaluate_validation: int = 1,
              tqdm_bar: bool = False,
              max_iterations_train: int = None,
              max_iterations_val: int = None,
              save_log: bool = True,
              *args, **kwargs) -> None:
        """
        performs a training session

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            train data loader
            Example : torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        train_eval_loader : torch.utils.data.DataLoader
            train data loader used for validation, if not given train_loader will be used
            Example : torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader : torch.utils.data.DataLoader, optional
            validation data loader, can be val_loader=None to skip validation evaluation
            Example : torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        train_epochs : int
            number of epochs to train
        optimizer_params : dict, optional
            optimizer params for the training session (default is {})
        prints : str, optional
            if prints=`display` displays epoch stats after each epoch, if prints=`print` prints epoch stats after each epoch
        callback_kwargs : dict, optional
            optional kwargs for self.callback
        device : str or torch.device
            device to be used for training,
            Example : >>> torch.device("cuda" if torch.cuda.is_available() else "cpu")
        save : bool, optional
            if save=True, saves the recent model checkpoint after each epoch and saves best model when new best is found (based on self.best_policy)
            if save=True and epochs_save>0, also saves model each epochs_save num of epochs (default is False)
        epochs_save : int, optional
            if save=True and epochs_save>0, saves model each epochs_save num of epochs (default is None)
        epochs_evaluate_train : int, optional
            if epochs_evaluate_train>0, evaluates model on the train set (in evaluation mode with no_grad) each epochs_evaluate_train num of epochs (default is 1)
        epochs_evaluate_validation : int, optional
            if epochs_evaluate_validation>0, evaluates model on the validation set (in evaluation mode with no_grad) each epochs_evaluate_validation num of epochs (default is 1)
        tqdm_bar : bool, optional
            if tqdm_bar=True, shows a tqdm progress bar over train_loader and val_loader (default is False)
        max_iterations_train : int, optional
            max train batches per epoch, will stop sampling from train_loader after max_iterations_train batches (default is None)
        max_iterations_val : int, optional
            max validation batches per epoch, will stop sampling from val_loader after max_iterations_val batches (default is None)
        save_log : bool, optional
            if save_log=True, saves train loss log for each training batch (default is True)
        *args, **kwargs
            additional args and kwargs will be passed to self.batch_pass
        """
        if train_epochs <= 0:
            return

        self.model = self.model.to(device)
        start_epoch = self.get_log()
        start_time = self.get_log('train_time')

        # if train_eval_loader was not given, use train_loader
        if train_eval_loader is None:
            train_eval_loader = train_loader

        # changing training hyperparams
        for param, val in optimizer_params.items():
            for group, _ in enumerate(self.optimizer.param_groups):
                self.optimizer.param_groups[group][param] = val

        # epochs loop
        tic = time.time()
        for train_epoch in range(train_epochs):
            epoch = train_epoch + start_epoch + 1

            # train run
            self.model.train()
            self.train_mode = True
            if self.seed is not None:
                with set_temp_seed(self.seed):
                    train_loss, train_score, train_results = self.run(
                        device, train_loader, epoch, train=True, tqdm_bar=tqdm_bar, max_iterations=max_iterations_train, *args, **kwargs)
            else:
                train_loss, train_score, train_results = self.run(
                    device, train_loader, epoch, train=True, tqdm_bar=tqdm_bar, max_iterations=max_iterations_train, *args, **kwargs)
            self.optimizer.zero_grad()

            # evaluation runs
            with torch.no_grad():
                self.model.eval()
                self.train_mode = False
                if epochs_evaluate_train is not None and epoch % epochs_evaluate_train == 0:
                    train_loss, train_score, train_results = self.run(
                        device, train_eval_loader, epoch, train=True, tqdm_bar=tqdm_bar, max_iterations=max_iterations_val, *args, **kwargs)

                if val_loader is not None and epochs_evaluate_validation is not None and epoch % epochs_evaluate_validation == 0:
                    val_loss, val_score, val_results = self.run(
                        device, val_loader, epoch, train=False, tqdm_bar=tqdm_bar, max_iterations=max_iterations_val, *args, **kwargs)
                else:
                    val_loss, val_score, val_results = None, None, {}

            # update self.log
            train_time = float(start_time + (time.time() - tic)/60)
            row = {
                'train_time': train_time,
                'timestamp': time.strftime('%H:%M:%S %d-%m-%Y'),
                'train_loss': train_loss,
                'val_loss': val_loss,
                # 'train_score': train_score,
                # 'val_score': val_score,
                'batch_size': self.batch_size,
            }
            row.update({'train_' + key: val for key,
                       val in train_results.items()})
            row.update({'val_' + key: val for key, val in val_results.items()})
            row.update(self._get_optimizer_params())
            row.update({key: val for key, val in callback_kwargs.items()})
            if row[self.best_policy] is not None:
                if self.best_policy in ['train_loss', 'val_loss']:
                    best = row[self.best_policy] < self.get_log(
                        self.best_policy, epoch='best')
                else:
                    best = row[self.best_policy] > self.get_log(
                        self.best_policy, epoch='best')
            else:
                best = False
            best = False
            row['best'] = best
            self.log = self.log.append(pd.Series(row, name=1 if len(
                self.log.index) == 0 else (max(self.log.index) + 1)), ignore_index=False)

            # callback and lr_scheduler step
            self.callback(**callback_kwargs)

            # save checkpoint
            if save:
                self.save(epoch=epoch)

            # epoch progress prints
            if prints == 'display':
                display(self.log.tail(1))
            elif prints == 'print':
                print('epoch {:3d}/{:3d} | train_loss {:.5f} | val_loss {:.5f} | train_time {:6.2f} min{:}'
                      .format(epoch, train_epochs + start_epoch, train_loss, val_loss, train_time, " | best" if best else ""))

    def predict(self,
                loader: torch.utils.data.DataLoader,
                device: torch.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"),
                tqdm_bar: bool = False,
                *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        returns a concatenation of raw batch_results, can be used for getting raw model predictions if implemented in batch_pass

        Parameters
        ----------
        device : str or torch.device
            device to be used for training,
            Example : >>> torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader : torch.utils.data.DataLoader
            data loader, must implement __iter__ and __len__ methods over batches
            Example : torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=False)
        tqdm_bar : bool, optional
            if tqdm_bar=True, shows a tqdm progress bar over loader

        Returns
        -------
            dict
                a concatenation of raw batch_results 
        """
        self.model = self.model.to(device)
        with torch.no_grad():
            self.model.eval()
            self.raw_results = {}

            if tqdm_bar:
                from tqdm import tqdm
                loader = tqdm(loader, total=len(loader))

            # mini-batch loop
            for batch in loader:
                # pass batch through model, calculate loss and predict
                _, batch_results = self.batch_pass(
                    device, batch, train=False, *args, **kwargs)

                # update evaluation results
                for key, val in batch_results.items():
                    if key not in self.raw_results:
                        self.raw_results[key] = list()
                    self.raw_results[key].append(val)

            for result in self.raw_results:
                self.raw_results[result] = np.concatenate(
                    self.raw_results[result])

            return self.raw_results

    def evaluate(self,
                 loader: torch.utils.data.DataLoader,
                 device: torch.device = torch.device(
                     "cuda" if torch.cuda.is_available() else "cpu"),
                 tqdm_bar: bool = False,
                 *args, **kwargs) -> Tuple[float, float, dict]:
        """
        returns a concatenation of raw batch_results, can be used for getting raw model predictions if implemented in batch_pass

        Parameters
        ----------
        device : str or torch.device
            device to be used for training,
            Example : >>> torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader : torch.utils.data.DataLoader
            data loader, must implement __iter__ and __len__ methods over batches
            Example : torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=False)
        tqdm_bar : bool, optional
            if tqdm_bar=True, shows a tqdm progress bar over loader

        Returns
        -------
            float
                model loss over loader
            float
                model score over loader
            dict
                results over loader
        """
        self.model = self.model.to(device)
        with torch.no_grad():
            self.model.eval()
            self.train_mode = False
            loss, score, results = self.run(
                device, loader, None, train=False, tqdm_bar=tqdm_bar, *args, **kwargs)
        return loss, score, results

    def summarize(self) -> None:
        """
        prints the graphs:
            * val_score, train_score
            * val_loss, train_loss
            * batch_size
            * lr graphs
        displays self.log
        """
        self.plot_checkpoint(['val_score', 'train_score'],
                             'score', 'score', scale='linear', base=10, save=False)
        self.plot_checkpoint(['val_loss', 'train_loss'],
                             'loss', 'loss', scale='linear', base=10, save=False)
        self.plot_checkpoint(['batch_size'], 'batch_size',
                             'batch_size', scale='log', base=2, save=False)
        self.plot_checkpoint(['lr'], 'lr', 'lr',
                             scale='log', base=10, save=False)
        display(self.log)
