import cv2
import numpy as np


def generate_heatmap(image, mask):

    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return overlay