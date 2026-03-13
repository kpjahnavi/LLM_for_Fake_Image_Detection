import torch
import io
from contextlib import redirect_stdout, redirect_stderr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_fast=False
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

model.eval()


def extract_visual_evidence(image_path):
    """
    Extracts detailed visual description for forensic reasoning.
    """

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,    # allow richer description
            num_beams=5
        )

    caption = processor.decode(output_ids[0], skip_special_tokens=True)

    return caption
