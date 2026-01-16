import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# 输入输出路径
input_dir = "data/dataset/images/"
output_dir = "data/dataset/masks_pseudo/"

os.makedirs(output_dir, exist_ok=True)

# 获取所有文件名，并按数字顺序排序
files = sorted(
    [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")],
    key=lambda x: int(x.replace("Image", "").replace(".jpg", ""))
)

for fname in files:
    print("Processing:", fname)

    # Step 1: 读取图像
    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: 计算灰度直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    # Step 3: 自动检测峰值
    peaks, _ = find_peaks(hist, distance=10, prominence=100)

    # Step 4: 找谷值
    valleys = []
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i+1]
        valley = np.argmin(hist[start:end]) + start
        valleys.append(valley)

    # Step 5: 选择最深谷值作为阈值
    valley_depths = [hist[peaks[i]] - hist[valleys[i]] for i in range(len(valleys))]
    best_valley_index = np.argmax(valley_depths)
    best_thresh = valleys[best_valley_index]

    # Step 6: 分割
    mask = (gray < best_thresh).astype(np.uint8) * 255
    inv = cv2.bitwise_not(mask)

    # Step 7: 填洞
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv)
    holes = np.zeros_like(mask)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if x > 0 and y > 0 and x + w < mask.shape[1] and y + h < mask.shape[0]:
            holes[labels == i] = 255

    mask_filled = cv2.bitwise_or(mask, holes)

    # Step 8: 保存 mask
    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, mask_filled)

print("全部处理完成！")
