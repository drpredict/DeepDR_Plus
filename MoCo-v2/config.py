import os
from src import utils



cfg = utils.Config()
cfg.moco = utils.Config()
cfg.clf = utils.Config()

###########################################################################
########################### general hyperparams ###########################
###########################################################################
cfg.seed = None
cfg.data_path = os.path.join('data')
cfg.models_dir = 'models'
cfg.save = True                     # save model checkpoints (best, last and epoch)
cfg.save_log = True                 # additionally save log and training loss logs to a .csv file
cfg.epochs_evaluate_train = 1       # evaluate train (in eval mode with no_grad) every epochs_evaluate_train epochs
cfg.epochs_evaluate_validation = 1  # evaluate validation (in eval mode with no_grad) every epochs_evaluate_validation epochs
cfg.num_workers = 2                 # num_workers for data loader
cfg.epochs_save = None              # save a checkpoint (additionally to last and best) every epochs_save epochs
cfg.tqdm_bar = False                # using a tqdm bar for loading data and epoch progression, should be False if not using a jupyter notebook
cfg.preload_data = False            # preloading data to memory
cfg.prints = 'print'                # should be 'display' if using a jupyter notebook, else 'print'


###########################################################################
############################# moco hyperparams ############################
###########################################################################
cfg.moco.load = -1
cfg.moco.wd = 1e-4
cfg.moco.backbone = 'resnet50'  # resnext50_32x4d resnet50 resnet34 wide_resnet50_2
cfg.moco.bs = 32  # 32 96 64
cfg.moco.temperature = 0.2  # 0.07 0.2
cfg.moco.queue_size = 16384  # 32768 16384 8192 int(cfg.moco.bs * 128)
cfg.moco.epochs = 600  # 600 800 1000

cfg.moco.optimizer_params = {}
cfg.moco.optimizer_momentum = 0.9
cfg.moco.lr = 1e-5
cfg.moco.min_lr = 5e-7
cfg.moco.cos = True
cfg.moco.best_policy = 'val_loss'
cfg.moco.model_momentum = 0.999
cfg.moco.dim = 128
cfg.moco.mlp = True
cfg.moco.bias = True
cfg.moco.clf_kwargs = {'random_state': 42, 'max_iter': 10000}
cfg.moco.train_transforms = utils.moco_v2_transforms
cfg.moco.train_eval_transforms = utils.TwoCropsTransform(utils.clf_train_transforms)
cfg.moco.val_eval_transforms = utils.TwoCropsTransform(utils.clf_val_transforms)
cfg.moco.version = f'{cfg.moco.backbone}_bs{cfg.moco.bs}_queue{cfg.moco.queue_size}_wd{cfg.moco.wd}_t{cfg.moco.temperature}{"_cos" if cfg.moco.cos else ""}'


###########################################################################
############################# clf hyperparams #############################
###########################################################################
cfg.clf.load = -1
cfg.clf.moco_epoch = 'best'
cfg.clf.epochs = 200
cfg.clf.wd = 0.0
cfg.clf.lr = 3e-2
cfg.clf.cos = True

cfg.clf.optimizer_params = {}
cfg.clf.version = cfg.moco.version + f'_epoch{cfg.clf.moco_epoch}_clf_wd{cfg.clf.wd}{"_cos" if cfg.clf.cos else ""}'
cfg.clf.best_policy = 'val_score'
cfg.clf.bs = cfg.moco.bs
cfg.clf.optimizer_momentum = 0.9
cfg.clf.min_lr = 5e-7
cfg.clf.train_transforms = utils.clf_train_transforms
cfg.clf.val_transforms = utils.clf_val_transforms



