import time

import torch
import torch.nn as nn

import torchvision




class Dummy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class MoCo_v2(nn.Module):
    """
    an implementation of MoCo v1 + v2

    MoCo v1: https://arxiv.org/abs/1911.05722
    MoCo v2: https://arxiv.org/abs/2003.04297
    """
    def __init__(self,
                 num_classes: int=10,
                 backbone: str='resnet50',
                 dim: int=256,
                 queue_size: int=65536,
                 batch_size: int=128,
                 momentum: float=0.999,
                 temperature: float=0.07,
                 bias: bool=True,
                 moco: bool=False,
                 clf_hyperparams: dict=dict(),
                 seed: int=42,
                 mlp: bool=True,  # MoCo v2 improvement
                 *args,
                 **kwargs,
                ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.dim = dim  # C
        self.queue_size = queue_size  # K
        self.batch_size = batch_size  # N
        self.momentum = momentum  # m
        self.temperature = temperature  # t
        self.bias = bias
        self.moco = moco
        self.clf_hyperparams = clf_hyperparams
        self.seed = seed
        if self.seed is not None: 
            torch.random.manual_seed(self.seed)

        self.mlp = mlp
        self.args = args
        self.kwargs = kwargs

        # create the queue
        self.register_buffer("queue",  nn.functional.normalize(torch.randn(self.queue_size, self.dim, requires_grad=False), dim=1)/10)
        self.ptr = 0

        # use requested torchvision backbone as a base encoder
        self.base_encoder = vars(torchvision.models)[self.backbone]

        # create and init query encoder
        self.q_encoder = self.base_encoder(num_classes=self.dim)
        self.k_encoder = self.base_encoder(num_classes=self.dim)
        if self.mlp:
            self.q_encoder.fc = nn.Sequential(nn.Linear(self.q_encoder.fc.weight.shape[1], self.q_encoder.fc.weight.shape[1]),
                                              nn.ReLU(),
                                              nn.Linear(self.q_encoder.fc.weight.shape[1], self.dim, bias=self.bias))
            self.k_encoder.fc = nn.Sequential(nn.Linear(self.k_encoder.fc.weight.shape[1], self.k_encoder.fc.weight.shape[1]),
                                              nn.ReLU(),
                                              nn.Linear(self.k_encoder.fc.weight.shape[1], self.dim, bias=self.bias))

        # init key encoder with query encoder weights
        self.k_encoder.load_state_dict(self.q_encoder.state_dict())

        # freeze k_encoder params (for manual momentum update)
        for p_k in self.k_encoder.parameters():
            p_k.requires_grad = False

        # detach the fc layers for accessing the fc input encodings
        self.q_fc = self.q_encoder.fc
        self.k_fc = self.k_encoder.fc
        self.q_encoder.fc = Dummy()
        self.k_encoder.fc = Dummy()

    def end_moco_phase(self):
        """ transition model to classification phase """
        self.moco = False

        # delete non-necessary modules and freeze all weights
        del self.k_encoder
        del self.queue
        del self.ptr
        del self.k_fc
        for p in self.parameters():
            p.requires_grad = False

        # init new fc encoder layer
        self.q_encoder.fc = nn.Linear(self.q_fc[0].weight.shape[1], self.num_classes, bias=self.bias)
        self.q_encoder.fc.weight.data = torch.FloatTensor(self.clf.coef_)
        self.q_encoder.fc.bias.data = torch.FloatTensor(self.clf.intercept_)

        # del sklearn classifier and old mlp/fc layer
        del self.q_fc
        try:
            del self.clf
        except:
            pass

        # make sure new fc layer grad enables
        for p in self.q_encoder.fc.parameters():
            p.requires_grad = True

    @torch.no_grad()
    def update_k_encoder_weights(self):
        """ manually update key encoder weights with momentum and no_grad"""
        # update k_encoder.parameters
        for p_q, p_k in zip(self.q_encoder.parameters(), self.k_encoder.parameters()):
            p_k.data = p_k.data*self.momentum + (1.0 - self.momentum)*p_q.data
            p_k.requires_grad = False

        # update k_fc.parameters
        for p_q, p_k in zip(self.q_fc.parameters(), self.k_fc.parameters()):
            p_k.data = p_k.data*self.momentum + (1.0 - self.momentum)*p_q.data
            p_k.requires_grad = False

    @torch.no_grad()
    def update_queue(self, k):
        """ swap oldest batch with the current key batch and update ptr"""
        self.queue[self.ptr: self.ptr + self.batch_size, :] = k.detach().cpu()
        self.ptr = (self.ptr + self.batch_size) % self.queue_size
        self.queue.requires_grad = False

    def forward(self, *args, prints=False):
        if self.moco:
            return self.moco_forward(*args, prints=prints)
        else:
            return self.clf_forward(*args, prints=prints)

    def moco_forward(self, q, k, prints=False):
        """ moco phase forward pass """
        print('q in', q.shape) if prints else None
        print('k in', k.shape) if prints else None

        q_enc = self.q_encoder(q)  # queries: NxC
        q = self.q_fc(q_enc)
        q = nn.functional.normalize(q, dim=1)
        print('q_encoder(q)', q.shape) if prints else None

        with torch.no_grad():
            k = self.k_encoder(k)  # keys: NxC
            k = self.k_fc(k)
            k = nn.functional.normalize(k, dim=1)
        print('k_encoder(k)', k.shape) if prints else None

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        print('l_pos', l_pos.shape) if prints else None

        # negative logits: NxK
        print('self.queue', self.queue.shape) if prints else None
        l_neg = torch.einsum('nc,kc->nk', [q, self.queue.clone().detach()])
        print('l_neg', l_neg.shape) if prints else None

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        print('logits', logits.shape) if prints else None

        # contrastive loss labels, positive logits used as ground truth
        zeros = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        print('zeros', zeros.shape) if prints else None

        self.update_k_encoder_weights()
        self.update_queue(k)

        return q_enc.detach(), logits, zeros

    def clf_forward(self, x, prints=False):
        """ clf phase forward pass """
        print('x in', x.shape) if prints else None

        x = self.q_encoder(x)
        print('q_encoder(x)', x.shape) if prints else None

        return x

    def print_hyperparams(self):
        return f'{self.backbone}_dim{self.dim}_queue_size{self.queue_size}_batch_size{self.batch_size}_momentum{self.momentum}_temperature{self.temperature}_{"mlp" if self.mlp else "no_mlp"}'


    
