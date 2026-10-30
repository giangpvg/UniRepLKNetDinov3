# UniRepLKNet with a DINOv3 backbone, adapted to the AST-style audio pipeline (dataloader.py / traintest.py).
# The repo root (unireplknet_backbone_dinov3.py, audio_loss.py) must be on sys.path, run.py takes care of it.
import torch
import torch.nn as nn

from unireplknet_backbone_dinov3 import UniRepLKNet
from audio_loss import make_frozen_teacher


class AudioDINOv3(nn.Module):
    """
    Input: the (B, time, freq) fbank of the dataloader, T and F must be multiples of 16.
    eval mode:  forward returns the logits (B, num_classes), like ASTModel.
    train mode: forward returns the dict of UniRepLKNet.forward_with_aux, plus 'teacher_tokens' (the patch tokens of a
                frozen copy of the pretrained backbone) when use_teacher is set. It is meant for AudioDINOLoss.
    """
    def __init__(self, num_classes, backbone_arch='dinov3_vits16', backbone_ckpt=None, drop_path_rate=0., use_teacher=True):
        super().__init__()
        self.net = UniRepLKNet(in_chans=1, num_classes=num_classes, backbone_arch=backbone_arch,
                               backbone_ckpt=backbone_ckpt, drop_path_rate=drop_path_rate)
        self.teacher = make_frozen_teacher(self.net) if use_teacher else None

    def train(self, mode=True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def forward(self, x):
        x = x.unsqueeze(1).transpose(2, 3)      # (B, T, F) -> (B, 1, F, T)
        if not self.training:
            return self.net(x)
        out = self.net.forward_with_aux(x)
        if self.teacher is not None:
            with torch.no_grad():
                out['teacher_tokens'] = self.teacher.forward_tokens(x)
        return out
