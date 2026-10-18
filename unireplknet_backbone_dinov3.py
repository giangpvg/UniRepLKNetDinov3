# UniRepLKNet with a DINOv3 backbone.
# Based on UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet), unireplknet.py.
# Only the backbone (stem, downsample layers and the 4 stages) is replaced by a DINOv3 ViT, the DINOv3 code is
# imported from the dinov3 repo, not copied. The final norm, the head, the output modes and the OpenMMLab init are kept.
import os
import sys
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# DINOv3 repo is expected at <UniRepLK>/dinov3, override with the DINOV3_REPO env var
sys.path.insert(0, os.environ.get('DINOV3_REPO', str(Path(__file__).resolve().parent / 'dinov3')))
from dinov3.hub import backbones as dinov3_backbones

has_mmseg = False
#   =============== for the ease of directly using this file in MMSegmentation.
#   =============== ignore the following segment of code if you do not plan to do so
try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmseg = True
except ImportError:
    get_root_logger = None
    _load_checkpoint = None
#   ===========================================================================================


class DINOv3Backbone(nn.Module):
    """
    DINOv3 ViT used as the backbone of UniRepLKNet. Takes (N, C, H, W) with H and W multiples of patch_size (16).
    forward() returns what the original stem + 4 stages returned: a list of 4 feature maps with strides 4, 8, 16, 32
    (all with embed_dim channels). The ViT only produces stride-16 maps, so 4 evenly spaced blocks are tapped and
    resampled to the 4 strides.
    forward_global() returns the (N, embed_dim) mean of the final patch tokens, without building the pyramid.
    """
    def __init__(self, arch='dinov3_vitb16', ckpt=None, drop_path_rate=0., in_chans=3):
        super().__init__()
        self.vit = getattr(dinov3_backbones, arch)(pretrained=ckpt is not None,
                                                   weights=ckpt if ckpt is not None else dinov3_backbones.Weights.LVD1689M)
        # the hub entries hard-code drop_path_rate=0, so set the (linearly increasing) stochastic depth here
        for i, blk in enumerate(self.vit.blocks):
            blk.sample_drop_ratio = drop_path_rate * i / max(len(self.vit.blocks) - 1, 1)
        self.embed_dim = self.vit.embed_dim
        depth = len(self.vit.blocks)
        self.out_indices = [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]
        #   DINOv3 is pretrained on RGB, so collapse the patch-embedding kernel to the requested channels
        old = self.vit.patch_embed.proj
        if old.in_channels != in_chans:
            new = nn.Conv2d(in_chans, old.out_channels, kernel_size=old.kernel_size, stride=old.stride)
            with torch.no_grad():
                new.weight.copy_(old.weight.sum(dim=1, keepdim=True).expand(-1, in_chans, -1, -1) / in_chans)
                new.bias.copy_(old.bias)
            self.vit.patch_embed.proj = new
            self.vit.patch_embed.in_chans = in_chans

    def forward_global(self, x):
        return self.vit.forward_features(x)['x_norm_patchtokens'].mean(dim=1)

    def forward(self, x):
        f1, f2, f3, f4 = self.vit.get_intermediate_layers(x, n=self.out_indices, reshape=True)
        return [F.interpolate(f1, scale_factor=4, mode='bilinear', align_corners=False),
                F.interpolate(f2, scale_factor=2, mode='bilinear', align_corners=False),
                f3,
                F.max_pool2d(f4, kernel_size=2, stride=2)]


class UniRepLKNet(nn.Module):
    r""" UniRepLKNet with a DINOv3 backbone

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head, None means no head and the forward returns the pooled feature vector. Default: 1000
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        init_cfg (dict): weights to load. The easiest way to use UniRepLKNet with for OpenMMLab family. Default: None
        backbone_arch (str): DINOv3 hub entry used as the backbone, e.g. dinov3_vits16, dinov3_vitb16, dinov3_vitl16. Default: dinov3_vitb16
        backbone_ckpt (str): path/URL of DINOv3 pretrained weights, None means random init. Default: None
        Other keyword arguments of the original UniRepLKNet (depths, dims, kernel_sizes, deploy, ...) are accepted and ignored.
    """
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 drop_path_rate=0.,
                 head_init_scale=1.,
                 init_cfg=None,
                 backbone_arch='dinov3_vitb16',
                 backbone_ckpt=None,
                 **kwargs
                 ):
        super().__init__()

        self.backbone = DINOv3Backbone(arch=backbone_arch, ckpt=backbone_ckpt, drop_path_rate=drop_path_rate,
                                       in_chans=in_chans)
        last_channels = self.backbone.embed_dim

        self.for_pretrain = init_cfg is None
        self.for_downstream = not self.for_pretrain     # there may be some other scenarios
        if self.for_downstream:
            assert num_classes is None

        if self.for_pretrain:
            self.init_cfg = None
            self.norm = nn.LayerNorm(last_channels, eps=1e-6)  # final norm layer
            #   num_classes=None: no classification head, forward returns the (N, embed_dim) feature vector
            self.head = nn.Linear(last_channels, num_classes) if num_classes is not None else nn.Identity()
            #   only the new head is initialized here, the backbone keeps its DINOv3 weights
            self._init_weights(self.head)
            if num_classes is not None:
                self.head.weight.data.mul_(head_init_scale)
                self.head.bias.data.mul_(head_init_scale)
            self.output_mode = 'logits'
        else:
            self.init_cfg = init_cfg        # OpenMMLab style init
            self.init_weights()
            self.output_mode = 'features'
            norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
            for i_layer in range(4):
                layer = norm_layer(last_channels)
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)


    #   load pretrained backbone weights in the OpenMMLab style
    def init_weights(self):

        def load_state_dict(module, state_dict, strict=False, logger=None):
            unexpected_keys = []
            own_state = module.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    unexpected_keys.append(name)
                    continue
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError(
                        'While copying the parameter named {}, '
                        'whose dimensions in the model are {} and '
                        'whose dimensions in the checkpoint are {}.'.format(
                            name, own_state[name].size(), param.size()))
            missing_keys = set(own_state.keys()) - set(state_dict.keys())

            err_msg = []
            if unexpected_keys:
                err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
            if missing_keys:
                err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
            err_msg = '\n'.join(err_msg)
            if err_msg:
                if strict:
                    raise RuntimeError(err_msg)
                elif logger is not None:
                    logger.warn(err_msg)
                else:
                    print(err_msg)

        logger = get_root_logger()
        assert self.init_cfg is not None
        ckpt_path = self.init_cfg['checkpoint']
        if ckpt_path is None:
            print('================ Note: init_cfg is provided but I got no init ckpt path, so skip initialization')
        else:
            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            load_state_dict(self, _state_dict, strict=False, logger=logger)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.output_mode == 'logits':
            x = self.backbone.forward_global(x)
            x = self.norm(x)
            x = self.head(x)
            return x
        elif self.output_mode == 'features':
            outs = self.backbone(x)
            return [self.__getattr__(f'norm{i}')(o) for i, o in enumerate(outs)]
        else:
            raise ValueError('Defined new output mode?')


class LayerNorm(nn.Module):
    r""" LayerNorm implementation used in ConvNeXt
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", reshape_last_to_first=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.reshape_last_to_first = reshape_last_to_first

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


#   For easy use as backbone in MMSegmentation framework. Ignore these lines if you do not use MMSegmentation
if has_mmseg:
    @seg_BACKBONES.register_module()
    class UniRepLKNetBackbone(UniRepLKNet):
        def __init__(self,
                     drop_path_rate=0.,
                     init_cfg=None,
                     backbone_arch='dinov3_vitb16',
                     backbone_ckpt=None,
                     **kwargs):
            assert init_cfg is not None
            super().__init__(in_chans=3, num_classes=None, drop_path_rate=drop_path_rate, init_cfg=init_cfg,
                             backbone_arch=backbone_arch, backbone_ckpt=backbone_ckpt)


if __name__ == '__main__':
    #   Smoke test: a 1-channel 128x1024 spectrogram goes in, a (N, embed_dim) vector comes out
    model = UniRepLKNet(in_chans=1, num_classes=None, backbone_arch='dinov3_vits16').eval()
    print(model(torch.rand(2, 1, 128, 1024)).shape)
