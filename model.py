
# %%
from collections import defaultdict
from typing import Any, Dict, Union, List, Tuple, Callable
from torch.nn import Module
import torch.nn as nn
from torchvision.models.resnet import resnet50, resnet18, resnet101
from torchvision.models import convnext_tiny
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torchvision.models.vision_transformer import vit_b_16
from abc import ABC, abstractclassmethod


class Hooks(Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.init(**kwargs)

    def init(self):
        pass

    def forward_hook_func(self, module: nn.Module, input: Any, output: Any) -> Any:
        raise NotImplementedError()


class AttentionHook(Hooks):
    def init(self, n_channels):
        self.query, self.key, self.value = (
            self._conv(n_channels, c)
            for c in (1, 1, n_channels)
        )
        self.gamma = nn.Parameter(torch.FloatTensor([0.]))
        self.last_attention = None

    def _conv(self, n_in, n_out):
        return nn.Conv1d(
            n_in,
            n_out,
            kernel_size=1,
            bias=False
        )

    def forward_hook_func(self, module: Module, input: Any, output: Any) -> Any:
        x = output
        size = x.size()
        x = x.view(*size[:2], -1)
        # X: B x C x WH
        f, g, h = self.query(x), self.key(x), self.value(x)
        # f: B C
        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), dim=1)
        o = (1-self.gamma) * torch.bmm(h, beta) + self.gamma * x
        o = o.view(*size).contiguous()
        self.last_attention = beta
        return o


@dataclass
class ModelDef():
    backbone: nn.Module
    output_size: int
    hooks: List[Tuple[str, Module, Hooks]]


def _backbone_resnet18(*args, **kwargs):
    model = resnet18(*args, **kwargs)
    model.fc = nn.Identity()
    return ModelDef(
        model,
        512,
        [
            ('attention', model.layer3, AttentionHook(n_channels=256))
        ],
    )


def _backbone_resnet50(*args, **kwargs):
    model = resnet50(*args, **kwargs)
    model.fc = nn.Identity()
    return ModelDef(
        model,
        2048,
        [
            ('attention', model.layer3, AttentionHook(n_channels=1024))
        ],
    )


class ModelProgression(Module):
    def __init__(self, backbone='resnet50', output_size=20, with_hooks=None, ** kwargs):
        super().__init__()
        model_def: ModelDef = globals(
        )[f'_backbone_{backbone}'](**kwargs)
        backbone = model_def.backbone
        feat_size = model_def.output_size
        self.hooks = model_def.hooks
        self.backbone = backbone
        self.drop_out = nn.Dropout()

        self.fc = nn.Sequential(
            nn.LayerNorm(feat_size, eps=1e-6, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(feat_size, output_size, bias=True),
        )
        # register hooks
        if with_hooks is None:
            with_hooks = set(['attention'])
        else:
            with_hooks = set(with_hooks)
        cnt_dict = defaultdict(lambda: 0)
        for hook_name, hook_module, hook in model_def.hooks:
            if hook_name in with_hooks:
                setattr(self, f'_hook_{hook_name}_{cnt_dict[hook_name]}', hook)
                if hook.__class__.forward_hook_func != Hooks.forward_hook_func:
                    hook_module.register_forward_hook(hook.forward_hook_func)
                cnt_dict[hook_name] = cnt_dict[hook_name] + 1

        self.forward_feat = {}
        self.attention_map = {}

    def forward(self, x):
        self.forward_feat.clear()
        self.attention_map.clear()
        feat = self.backbone(x)
        feat = feat.view(feat.shape[0], -1)
        feat = self.drop_out(feat)
        out = self.fc(feat)
        return out


if __name__ == "__main__":
    m = ModelProgression(backbone='resnet50', with_hooks=['attention'])
    output = m(torch.randn(3, 3, 512, 512))
    print(output.shape)

# %%
