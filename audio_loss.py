# Custom loss for UniRepLKNet-DINOv3 on audio spectrograms. No plain CE / BCE:
#   total = AsymmetricFocalLoss(logits, targets)                      classification, soft-label aware
#         + koleo_weight * KoLeoLoss(embed)                           DINOv3: spread the pooled embeddings
#         + gram_weight  * GramLoss(tokens, frozen_teacher_tokens)    DINOv3: keep the patch-token structure (optional)
import copy

import torch
import torch.nn as nn

import unireplknet_backbone_dinov3  # noqa: F401  (puts the dinov3 repo on sys.path)
from dinov3.loss import GramLoss, KoLeoLoss


class AsymmetricFocalLoss(nn.Module):
    """
    Sigmoid focal loss with separate focusing for positives and negatives, a probability margin on the negatives and
    label smoothing. Works with soft targets in [0, 1] (mixup, multi-hot), so it is a drop-in for BCEWithLogits.
        gamma_pos: focus on hard positives (0 = plain BCE term)
        gamma_neg: down-weights easy negatives, which dominate when most of the classes are absent
        margin:    negatives with p < margin contribute nothing (hard thresholding of very easy negatives)
        smoothing: targets become t * (1 - smoothing) + smoothing / 2
    """
    def __init__(self, gamma_pos=1., gamma_neg=3., margin=0.05, smoothing=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos, self.gamma_neg, self.margin, self.smoothing, self.eps = gamma_pos, gamma_neg, margin, smoothing, eps

    def forward(self, logits, targets):
        logits, targets = logits.float(), targets.float()
        targets = targets * (1 - self.smoothing) + self.smoothing / 2
        p = torch.sigmoid(logits)
        p_neg = (p - self.margin).clamp(min=0)
        loss_pos = targets * (1 - p).pow(self.gamma_pos) * torch.log(p.clamp(min=self.eps))
        loss_neg = (1 - targets) * p_neg.pow(self.gamma_neg) * torch.log((1 - p_neg).clamp(min=self.eps))
        return -(loss_pos + loss_neg).sum(dim=-1).mean()


class AudioDINOLoss(nn.Module):
    """
    Use with UniRepLKNet.forward_with_aux(). `teacher_tokens` are the patch tokens of a frozen copy of the pretrained
    backbone on the same input (see make_frozen_teacher); leave it None to disable the Gram term.
    Returns (total, dict of the detached components) so every term can be logged.
    """
    def __init__(self, gamma_pos=1., gamma_neg=3., margin=0.05, smoothing=0.05, koleo_weight=0.1, gram_weight=1.):
        super().__init__()
        self.cls_loss = AsymmetricFocalLoss(gamma_pos, gamma_neg, margin, smoothing)
        self.koleo = KoLeoLoss()
        self.gram = GramLoss(apply_norm=True, remove_neg=True)
        self.koleo_weight, self.gram_weight = koleo_weight, gram_weight

    def forward(self, out, targets, teacher_tokens=None):
        parts = {'cls': self.cls_loss(out['logits'], targets)}
        if self.koleo_weight > 0 and out['embed'].shape[0] > 1:
            parts['koleo'] = self.koleo_weight * self.koleo(out['embed'])
        if self.gram_weight > 0 and teacher_tokens is not None:
            parts['gram'] = self.gram_weight * self.gram(out['tokens'], teacher_tokens)
        total = sum(parts.values())
        return total, {k: v.detach() for k, v in parts.items()}


def make_frozen_teacher(model):
    """Frozen copy of model.backbone, call it right after the pretrained weights are loaded and before training."""
    teacher = copy.deepcopy(model.backbone).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    return teacher


if __name__ == '__main__':
    #   Smoke test: one train step on fake 128x1024 spectrograms with mixup-style soft labels
    from unireplknet_backbone_dinov3 import UniRepLKNet
    model = UniRepLKNet(in_chans=1, num_classes=35, backbone_arch='dinov3_vits16')
    teacher = make_frozen_teacher(model)
    criterion = AudioDINOLoss()
    x = torch.randn(4, 1, 128, 1024)
    y = torch.zeros(4, 35)
    y[torch.arange(4), torch.arange(4)] = 0.6
    y[torch.arange(4), torch.arange(4) + 10] = 0.4
    with torch.no_grad():
        t_tokens = teacher.forward_tokens(x)
    loss, parts = criterion(model.forward_with_aux(x), y, t_tokens)
    loss.backward()
    print('total', loss.item(), {k: round(v.item(), 4) for k, v in parts.items()})
    print('grad on head:', model.head.weight.grad.abs().sum().item() > 0,
          '| grad on vit:', model.backbone.vit.blocks[0].attn.qkv.weight.grad.abs().sum().item() > 0,
          '| teacher frozen:', all(not q.requires_grad for q in teacher.parameters()))
