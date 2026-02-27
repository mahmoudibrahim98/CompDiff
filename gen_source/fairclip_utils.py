"""
FairCLIP integration for the diffusion pipeline.

Option 1: Use a frozen FairCLIP model to add a fairness regularizer loss so that
image–text similarity (as measured by FairCLIP) is more balanced across demographic groups.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# Optional imports: FairCLIP uses openai/clip and geomloss
try:
    import clip
except ImportError:
    clip = None
try:
    from geomloss import SamplesLoss
except ImportError:
    SamplesLoss = None


def load_fairclip_model(
    checkpoint_path: str,
    device: torch.device,
    arch: str = "ViT-B/16",
    logger=None,
) -> Tuple[Optional["torch.nn.Module"], Optional[callable]]:
    """
    Load FairCLIP model and preprocess from a training checkpoint.

    Checkpoint should be a .pth saved by FairCLIP (finetune_FairCLIP.py) with
    key 'model_state_dict' (full CLIP image + text encoder).

    Args:
        checkpoint_path: Path to .pth file.
        device: Device to load the model on.
        arch: CLIP architecture string, e.g. "ViT-B/16" or "ViT-L/14".
        logger: Optional logger for messages.

    Returns:
        (model, preprocess) or (None, None) if clip/checkpoint unavailable.
    """
    if clip is None:
        if logger:
            logger.warning("FairCLIP regularizer skipped: 'clip' package not installed.")
        return None, None
    if not checkpoint_path or not torch.cuda.is_available() and device.type == "cuda":
        return None, None
    try:
        model, preprocess = clip.load(arch, device=device, jit=False)
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        if logger:
            logger.info(f"Loaded FairCLIP from {checkpoint_path} (arch={arch})")
        return model, preprocess
    except Exception as e:
        if logger:
            logger.warning(f"Could not load FairCLIP from {checkpoint_path}: {e}")
        return None, None


def compute_fairclip_fairness_loss(
    images: torch.Tensor,
    attr_indices: torch.Tensor,
    model: "torch.nn.Module",
    preprocess: callable,
    device: torch.device,
    default_text: str = "A fundus image",
    sinkhorn_blur: float = 1e-4,
    loss_fn: Optional["SamplesLoss"] = None,
) -> torch.Tensor:
    """
    Compute FairCLIP-style fairness loss: Sinkhorn distance between the batch
    distribution of image–text similarities and each demographic group's distribution.

    Args:
        images: [B, C, H, W] in [0, 1] (e.g. 512x512). Will be resized and preprocessed for CLIP.
        attr_indices: [B] integer group indices (e.g. race_idx or sex_idx).
        model: Frozen FairCLIP model (from load_fairclip_model).
        preprocess: CLIP preprocess (resize 224, normalize).
        device: Device.
        default_text: Text for FairCLIP text encoder (single prompt for all).
        sinkhorn_blur: Blur for Sinkhorn loss.
        loss_fn: Reusable SamplesLoss instance; if None, one is created.

    Returns:
        Scalar loss (mean Sinkhorn distance across groups).
    """
    if SamplesLoss is None:
        return torch.tensor(0.0, device=device, dtype=images.dtype)
    B = images.shape[0]
    # Resize to 224x224 and normalize with CLIP's mean/std (images assumed in [0, 1])
    if images.shape[2] != 224 or images.shape[3] != 224:
        images_224 = F.interpolate(
            images.float(),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
    else:
        images_224 = images.float()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
    images_clip = (images_224 - mean) / (std + 1e-8)
    images_clip = images_clip.to(device)

    with torch.no_grad():
        text_tokens = clip.tokenize([default_text], truncate=True).to(device)
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    img_feat = model.encode_image(images_clip)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    # Per-sample similarity: [B]
    sims = (img_feat @ text_feat.T).squeeze(-1).float()

    # Normalize batch similarities to a distribution (positive, sum=1)
    sims_min = sims.min().item()
    if sims_min <= 0:
        sims = sims - sims_min + 1e-6
    sims_batch = sims / (sims.sum() + 1e-12)
    if loss_fn is None:
        loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=sinkhorn_blur)

    total_loss = torch.tensor(0.0, device=device, dtype=sims.dtype)
    groups = attr_indices.unique()
    for g in groups:
        g = g.item()
        mask = (attr_indices == g).nonzero(as_tuple=True)[0]
        if mask.numel() == 0:
            continue
        sims_g = sims[mask]
        sims_g = sims_g - sims_g.min() + 1e-6
        sims_g = sims_g / (sims_g.sum() + 1e-12)
        # [B, 1] and [G, 1] for geomloss
        a = sims_batch.unsqueeze(1)
        b = sims_g.unsqueeze(1)
        total_loss = total_loss + loss_fn(a, b)
    n = max(1, len(groups))
    return total_loss / n
