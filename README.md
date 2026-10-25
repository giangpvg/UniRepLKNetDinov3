# UniRepLKNet with DINOv3 Backbone

A system integrating DINOv3 (Meta AI Research Vision Transformer) as the feature extraction backbone to replace the Large-Kernel CNN stages in the UniRepLKNet architecture.

---

## 1. Project Overview

This system integrates the foundation model **DINOv3 (Vision Transformer)** as the feature extraction backbone, replacing the entire large convolutional block (Stem and 4 Large-Kernel CNN Stages) of the UniRepLKNet architecture. 
The project provides a multimodal visual representation extraction solution (RGB images, Audio Spectrograms) capable of generating multi-scale Feature Pyramids and performing global representation classification.

## 2. Processing Pipeline and Component Roles
- **Execution Environment:** Runs locally (or on a GPU/CPU compute node).
- **AI Model Invocation:** The entire inference/feature extraction process is encapsulated within the `UniRepLKNet` class in [`unireplknet_backbone_dinov3.py`](./unireplknet_backbone_dinov3.py).
- **Feature Pyramid Reconstruction Mechanism:** Since ViTs inherently process tokens at a fixed resolution (stride 16), 4 evenly-spaced intermediate layers are tapped and resampled (bilinear interpolation $\times4$, $\times2$, identity, and max-pooling) to reconstruct feature maps with standard strides of 4, 8, 16, and 32, compatible with downstream heads.

---

## 3. Environment

The project requires a Python and PyTorch environment with the following prerequisites:

| Component | Version / Requirement | Notes |
| :--- | :--- | :--- |
| **Operating System (OS)** | Linux (Ubuntu 20.04 / 22.04 LTS), Windows 10/11, macOS | Verified stable on Ubuntu 22.04 LTS |
| **Python** | `3.10.x` (supports $\ge 3.9$) | Python 3.10 recommended |
| **PyTorch** | `torch >= 2.0.0`, `torchvision >= 0.15.0` | Verified on `torch 2.9.1+cu128` |
| **Hardware** | Minimum: 4-core CPU, 8 GB RAM | Recommended: NVIDIA GPU with $\ge$ 8GB VRAM for training / fine-tuning |
| **Model Weights** | DINOv3 LVD-1689M Weights | Loaded automatically via DINOv3 hub or local `.pth` checkpoint |
| **Downstream (Optional)** | `mmcv >= 2.0.0`, `mmsegmentation >= 1.0.0` | Required when used as a backbone in MMSegmentation |

---

## 4. Setup

Execute the following steps sequentially in your terminal:

### Step 1: Clone the repository
Run in your working directory:
```bash
git clone https://github.com/giangpvg/UniRepLKNetDinov3.git
cd UniRepLKNetDinov3
```
*Expected result:* The entire project codebase is downloaded, containing the `dinov3/` and `UniRepLKNet/` directories as well as `unireplknet_backbone_dinov3.py`.

### Step 2: Create and activate a virtual environment
Run in the repository root directory:
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (Command Prompt / PowerShell):
python -m venv .venv
.venv\Scripts\activate
```
*Expected result:* The command prompt displays the `(.venv)` prefix.

### Step 3: Install dependencies
Run the installation command using [`requirements.txt`](./requirements.txt):
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*Expected result:* The installation completes without errors; `torch`, `torchvision`, `omegaconf`, and `scikit-learn` packages are ready.

### Step 4: Verify the environment
```bash
python3 -c "import torch; print('PyTorch version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available())"
```
*Expected result:* Prints the PyTorch version (e.g., `PyTorch version: 2.x.x | CUDA Available: True/False`).

---

## 5. Configuration file (.env)

The system supports configuring the DINOv3 source path and checkpoint files via environment variables.

Create the `.env` file from the template [`.env.example`](./.env.example):
```bash
cp .env.example .env
```

Description of configuration variables:

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `DINOV3_REPO` | `dinov3` | Relative or absolute path to the DINOv3 source code directory |
| `BACKBONE_CKPT` | `None` *(optional)* | Path to a local pretrained `.pth` weight checkpoint file |

> **Security & Git Management Principles:**
> - The `.env.example` file is committed to Git as a reference template.
> - The `.env` file containing local paths is ignored via `.gitignore` and must not be committed to the repository.

---

## 6. Run & Verify

### 6.1. Run Default Smoke Test
Run `unireplknet_backbone_dinov3.py` directly from the root directory:
```bash
python3 unireplknet_backbone_dinov3.py
```

**Expected output:**
```text
torch.Size([2, 384])
```
*Explanation:* Simulated input data of 2 single-channel spectrogram samples sized $128 \times 1024$ passes through the `dinov3_vits16` network (embed_dim=384) and returns a standard feature vector of size `[2, 384]`.

---

### 6.2. Multi-mode Verification Test
Run the test script to simultaneously verify both **Classification** and **Multi-scale Feature Pyramid** modes:
```bash
python3 -c "
import torch
from unireplknet_backbone_dinov3 import UniRepLKNet

# Test 1: RGB Image Classification (1000 classes)
model_cls = UniRepLKNet(in_chans=3, num_classes=1000, backbone_arch='dinov3_vits16').eval()
out_cls = model_cls(torch.randn(2, 3, 224, 224))
print('Test 1 - Classification shape:', out_cls.shape)

# Test 2: Multi-Scale Feature Pyramid Extraction (Downstream)
model_seg = UniRepLKNet(in_chans=3, num_classes=None, init_cfg={'checkpoint': None}, backbone_arch='dinov3_vits16').eval()
outs = model_seg(torch.randn(2, 3, 224, 224))
print('Test 2 - Feature Pyramid shapes:', [o.shape for o in outs])
"
```

**Expected output:**
```text
Test 1 - Classification shape: torch.Size([2, 1000])
================ Note: init_cfg is provided but I got no init ckpt path, so skip initialization
Test 2 - Feature Pyramid shapes: [torch.Size([2, 384, 56, 56]), torch.Size([2, 384, 28, 28]), torch.Size([2, 384, 14, 14]), torch.Size([2, 384, 7, 7])]
```
*Explanation:*
- Branch 1: Classification output correctly returns `[2, 1000]` for batch size 2.
- Branch 2: Accurately reconstructs 4 resolution stages corresponding to stride 4 ($56\times56$), stride 8 ($28\times28$), stride 16 ($14\times14$), and stride 32 ($7\times7$).

---

## 7. AI Model Details

### 7.1. Model Specifications Table

| Property | Details |
| :--- | :--- |
| **Model Name** | `UniRepLKNet` with `DINOv3Backbone` |
| **Backbone Architectures** | `dinov3_vits16` (384-dim), `dinov3_vitb16` (768-dim), `dinov3_vitl16` (1024-dim) |
| **Pretrained Dataset** | Meta AI `LVD-1689M` (Self-Supervised Pretraining) |
| **Patch Size** | $16 \times 16$ pixels |
| **Tasks** | - Multimodal Classification (Image / Audio Spectrogram Classification)<br>- Multi-Scale Feature Extraction for Semantic Segmentation (MMSegmentation) & Object Detection |
| **Input Requirements** | - 4D Tensor of shape `(N, C, H, W)`<br>- Channel `C`: Customizable (`3` for color RGB images, `1` for grayscale images or audio spectrograms)<br>- Spatial dimensions $H, W$: **Must be multiples of 16** (e.g., $224\times224, 512\times512, 128\times1024$) |
| **Output Modes** | - `output_mode='logits'`: Returns tensor `(N, num_classes)` or `(N, embed_dim)`<br>- `output_mode='features'`: Returns a list of 4 tensors `[f1, f2, f3, f4]` with shapes $[H/4, W/4]$, $[H/8, W/8]$, $[H/16, W/16]$, and $[H/32, W/32]$ |
| **Regularization** | Stochastic Depth (`drop_path_rate`) linearly increasing from 0 to the specified rate across transformer blocks |

### 7.2. Current System Limitations
1. **Input Dimension Constraints:** Spatial dimensions $H$ and $W$ of input data must be divisible by 16. If not, resizing or padding is required prior to input.
2. **Memory Footprint:** Because ViT's global self-attention mechanism does not use window partitions (like Swin Transformer), GPU VRAM usage grows quadratically when input image dimensions scale up (e.g., $>1024\times1024$).
3. **Channel Adaptation:** When changing input channels to $C \ne 3$, patch projection weights are initialized by averaging RGB channels. Fine-tuning for a few epochs on the target dataset is recommended for optimal performance.

---

## 8. Usage & API

### 8.1. RGB Image Classification
```python
import torch
from unireplknet_backbone_dinov3 import UniRepLKNet

# Initialize ImageNet (1000 classes) classification model with DINOv3 Small backbone
model = UniRepLKNet(
    in_chans=3,
    num_classes=1000,
    backbone_arch='dinov3_vits16'
).eval()

# Dummy input image: Batch size 1, 3 RGB channels, size 224x224
dummy_image = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    logits = model(dummy_image)

print("Output shape:", logits.shape)
# Expected Output: torch.Size([1, 1000])
```

---

### 8.2. Audio Mel-Spectrogram Classification / Feature Extraction (`in_chans=1`)
```python
import torch
from unireplknet_backbone_dinov3 import UniRepLKNet

# Initialize model for audio: 1 spectrogram channel, no classification head
model = UniRepLKNet(
    in_chans=1,
    num_classes=None,
    backbone_arch='dinov3_vitb16'
).eval()

# Dummy spectrogram data: Batch size 4, 1 channel, 128 mel bins x 1024 time steps
dummy_spectrogram = torch.randn(4, 1, 128, 1024)

with torch.no_grad():
    features = model(dummy_spectrogram)

print("Audio representation vector shape:", features.shape)
# Expected Output: torch.Size([4, 768])
```

---

### 8.3. Multi-Scale Feature Pyramid Extraction for Segmentation (MMSegmentation / Downstream)
```python
import torch
from unireplknet_backbone_dinov3 import UniRepLKNet

# Initialize multi-level feature extraction mode (OpenMMLab style)
model = UniRepLKNet(
    in_chans=3,
    num_classes=None,
    init_cfg={'checkpoint': None},
    backbone_arch='dinov3_vits16'
).eval()

# Input image of size 512x512
input_tensor = torch.randn(1, 3, 512, 512)

with torch.no_grad():
    stages = model(input_tensor)

for i, stage_feat in enumerate(stages):
    print(f"Stage {i+1} Feature Map shape:", stage_feat.shape)

# Expected Output:
# Stage 1 Feature Map shape: torch.Size([1, 384, 128, 128])  (Stride 4)
# Stage 2 Feature Map shape: torch.Size([1, 384, 64, 64])    (Stride 8)
# Stage 3 Feature Map shape: torch.Size([1, 384, 32, 32])    (Stride 16)
# Stage 4 Feature Map shape: torch.Size([1, 384, 16, 16])    (Stride 32)
```

---

### 8.4. Integration Configuration for MMSegmentation
When configuring the backbone in an MMSegmentation config file:
```python
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='UniRepLKNetBackbone',
        in_chans=3,
        backbone_arch='dinov3_vitb16',
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint='path/to/dinov3_weights.pth')
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        channels=512,
        num_classes=150
    )
)
```

---

## 9. Troubleshooting

| Error Symptom | Possible Cause | Solution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'dinov3'` | Python cannot locate the `dinov3` directory due to environment variables or working directory | Verify that the `dinov3/` subfolder exists in the project or execute `export DINOV3_REPO=$(pwd)/dinov3` prior to running |
| `RuntimeError: The size of tensor a must match the size of tensor b` | Image dimensions $H$ or $W$ are not divisible by 16 (stride of patch embedding) | Preprocess input images via resizing or zero-padding to make $H, W$ multiples of 16 (e.g., $224, 256, 512, 1024$) |
| `CUDA out of memory` | ViT Base/Large model requires substantial VRAM when batch size or image resolution is too large | Reduce batch size to `1` or `2`, switch to a lighter variant such as `dinov3_vits16`, or use `torch.cuda.amp.autocast()` |
| `ImportError: cannot import name 'BACKBONES'` | `mmcv` and `mmsegmentation` packages are not installed when importing `UniRepLKNetBackbone` | If using pure PyTorch classification or feature extraction, use the `UniRepLKNet` class. If MMSegmentation is required, install via `mim install mmsegmentation mmcv` |
| `TypeError: 'NoneType' object is not callable` | Calling `init_weights` without an initialized `mmseg` logger | Automatically handled in the updated code; if invoked manually, ensure a valid `init_cfg` is provided |

---

## 10. References & Acknowledgements

This project is built upon and adapted from open-source repositories:
1. **[UniRepLKNet (AILab-CVC)](https://github.com/AILab-CVC/UniRepLKNet):** *A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition*. Reused the classification head structure, channels-first LayerNorm, multi-level downstream formats, and OpenMMLab integration interface.
2. **[DINOv3 (Meta AI Research)](https://github.com/facebookresearch/dinov3):** Leveraged the Vision Transformer (ViT) architecture and large-scale self-supervised pretrained weights on the `LVD-1689M` dataset.
