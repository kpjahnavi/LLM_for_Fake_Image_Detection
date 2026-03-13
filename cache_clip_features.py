import os
import torch
from torchvision.datasets import ImageFolder

from preprocessing.preprocess import preprocess_image
from feature_extraction.clip_encoder import extract_clip_features

# -----------------------------
# CONFIG
# -----------------------------
TRAIN_DIR = "dataset/train"
SAVE_DIR = "cached_features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset = ImageFolder(TRAIN_DIR)
print(f"Total images to cache: {len(dataset.samples)}")

features_list = []
labels_list = []

# -----------------------------
# FEATURE EXTRACTION (ONCE)
# -----------------------------
for idx, (img_path, label) in enumerate(dataset.samples):

    if idx % 100 == 0:
        print(f"Caching image {idx}/{len(dataset.samples)}")

    img_tensor = preprocess_image(img_path)

    with torch.no_grad():
        features = extract_clip_features(img_tensor)

    features_list.append(features.cpu())
    labels_list.append(label)

# -----------------------------
# SAVE FEATURES
# -----------------------------
features_tensor = torch.stack(features_list)   # (N, 768)
labels_tensor = torch.tensor(labels_list)      # (N,)

torch.save(
    {
        "features": features_tensor,
        "labels": labels_tensor
    },
    os.path.join(SAVE_DIR, "train_features.pt")
)

print("✅ CLIP feature caching completed successfully!")