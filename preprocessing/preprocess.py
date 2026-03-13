from PIL import Image
from torchvision import transforms

# CLIP-specific preprocessing
clip_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

def preprocess_image(image_path):
    """
    Input  : path to image
    Output : preprocessed image tensor (3, 224, 224)
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = clip_preprocess(image)
    return image_tensor
