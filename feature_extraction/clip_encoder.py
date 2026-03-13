import torch
import io
from contextlib import redirect_stdout, redirect_stderr
from transformers import CLIPModel
import warnings

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Use 768-dim CLIP so it matches the trained classifier input size.
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
clip_model.eval()


def _to_feature_tensor(clip_output):
    """
    Normalize API differences across transformers versions.
    """
    if isinstance(clip_output, torch.Tensor):
        return clip_output

    # Newer transformers can return BaseModelOutputWithPooling here.
    if hasattr(clip_output, "pooler_output") and clip_output.pooler_output is not None:
        return clip_output.pooler_output

    # Fallback for wrapper outputs that expose image_embeds directly.
    if hasattr(clip_output, "image_embeds") and clip_output.image_embeds is not None:
        return clip_output.image_embeds

    raise TypeError(
        f"Unsupported CLIP output type from get_image_features: {type(clip_output)}"
    )


def extract_clip_features(image_tensor):
    """
    Returns a 768-dim CLIP image embedding (torch.Tensor)
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        raw_output = clip_model.get_image_features(pixel_values=image_tensor)
        features = _to_feature_tensor(raw_output)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze(0)  # (768,)
