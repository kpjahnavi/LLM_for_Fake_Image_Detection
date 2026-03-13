import torch
import os
import cv2
import numpy as np

from preprocessing.preprocess import preprocess_image
from feature_extraction.clip_encoder import extract_clip_features
from localization.attention_localization import generate_attention_heatmap
from classification.classifier import FakeImageClassifier

# Explainability modules
from explainability.blip_explainer import extract_visual_evidence
from explainability.heatmap_analyzer import summarize_heatmap
from explainability.llm_reasoner import llm_reasoning


# --------------------------------------------------
# Load trained classifier (768-dim – CLEAN)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "fake_image_classifier.pth")

classifier = FakeImageClassifier()
state_dict = torch.load(MODEL_PATH, map_location="cpu")
classifier.load_state_dict(state_dict)
classifier.eval()


def predict_image(image_path):
    """
    End-to-end fake image detection pipeline
    (768-dim trained model – STABLE VERSION)
    """

    # -----------------------------
    # 1. Preprocessing
    # -----------------------------
    image_tensor = preprocess_image(image_path)   # (3, 224, 224)

    # -----------------------------
    # 2. CLIP Feature Extraction
    # -----------------------------
    features = extract_clip_features(image_tensor)   # (768,)
    features = features.unsqueeze(0)                  # (1, 768)

    # -----------------------------
    # 3. Classification
    # -----------------------------
    with torch.no_grad():
        logits = classifier(features)                # (1, 2)
        probs = torch.softmax(logits, dim=1)

        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

    prediction = "Real" if predicted_class == 0 else "Fake"

    # -----------------------------
    # 4. Attention Heatmap (SAFE)
    # -----------------------------
    heatmap_path = None
    try:
        heatmap = generate_attention_heatmap(image_tensor)

        heatmap_path = image_path.replace("uploads", "heatmaps")
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)

        heatmap_img = (heatmap * 255).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
        cv2.imwrite(heatmap_path, heatmap_img)

    except Exception as e:
        print(f"[WARN] Heatmap generation failed: {e}")

    # -----------------------------
    # 5. BLIP Visual Evidence
    # -----------------------------
    try:
        visual_evidence = extract_visual_evidence(image_path)
    except Exception as e:
        visual_evidence = "Visual evidence unavailable."
        print(f"[WARN] BLIP failed: {e}")

    # -----------------------------
    # 6. Heatmap Summary
    # -----------------------------
    try:
        heatmap_summary = summarize_heatmap(heatmap_path)
    except Exception as e:
        heatmap_summary = "Heatmap summary unavailable."
        print(f"[WARN] Heatmap summary failed: {e}")

    # -----------------------------
    # 7. LLM Reasoning
    # -----------------------------
    try:
        explanation_text = llm_reasoning(
            prediction=prediction,
            confidence=confidence,
            visual_evidence=visual_evidence,
            heatmap_summary=heatmap_summary
        )
    except Exception as e:
        explanation_text = (
            f"The image was classified as {prediction} "
            f"with confidence {confidence:.2f}."
        )
        print(f"[WARN] LLM reasoning failed: {e}")

    return prediction, confidence, heatmap_path, explanation_text
