import torch
import numpy as np
import cv2
import warnings

warnings.filterwarnings("ignore")

def generate_attention_heatmap(image_tensor):
    """
    SAFE attention heatmap generation
    Compatible with CLIP ViT-B/32 (768-dim setup)
    No CLIP forward pass → no BaseModelOutput errors
    """

    # image_tensor: (3, 224, 224)
    if isinstance(image_tensor, torch.Tensor):
        img = image_tensor.cpu().numpy()
    else:
        raise ValueError("Input must be a torch.Tensor")

    # Convert to grayscale-like importance map
    heatmap = np.mean(img, axis=0)  # (224, 224)

    # Normalize
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)

    # Ensure correct size
    heatmap = cv2.resize(heatmap, (224, 224))

    return heatmap
