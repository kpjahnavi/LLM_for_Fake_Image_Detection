import cv2
import numpy as np

def summarize_heatmap(heatmap_path):
    """
    Converts heatmap into a more descriptive textual summary.
    """

    heatmap = cv2.imread(heatmap_path, 0)
    avg_intensity = np.mean(heatmap)
    max_intensity = np.max(heatmap)

    if max_intensity > 220:
        return "Strong attention on specific localized regions such as edges and textured areas."
    elif avg_intensity > 140:
        return "Moderate attention distributed across multiple regions including foreground and background."
    else:
        return "Low overall attention with weak localization of suspicious regions, suggesting subtle artifacts."
