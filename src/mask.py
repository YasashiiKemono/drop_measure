# 背景差分+形态学清理
import cv2
import numpy as np

DIFF_THRESH = 20
KERNEL_SIZE = 5

def get_droplet_mask(frame_gray, background_gray):
    diff = cv2.absdiff(frame_gray, background_gray)
    _, mask = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask
