import numpy as np
import cv2
import matplotlib.pyplot as plt


def overlay_mask(image, mask, alpha=0.5, rgb=[255, 255, 0]):
    overlay = image.copy()
    overlay[mask == 255] = np.array(rgb, dtype=np.uint8)

    output = image.copy()
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output

def show_mask(mask):
    plt.figure()
    plt.imshow(mask, cmap='viridis')
