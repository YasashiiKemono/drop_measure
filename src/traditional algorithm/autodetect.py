import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Step 1: 读取图像并转换为灰度图
img = cv2.imread("data/dataset/images/Image23.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: 计算灰度直方图
hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

# Step 3: 自动检测所有峰值
peaks, _ = find_peaks(hist, distance=10, prominence=100)

# Step 4: 在每对相邻峰之间寻找谷值（最小值点）
valleys = []
for i in range(len(peaks) - 1):
    start, end = peaks[i], peaks[i+1]
    valley = np.argmin(hist[start:end]) + start
    valleys.append(valley)

# Step 5: 选择最深谷值作为最佳阈值
valley_depths = [hist[peaks[i]] - hist[valleys[i]] for i in range(len(valleys))]
best_valley_index = np.argmax(valley_depths)
best_thresh = valleys[best_valley_index]

# Step 6: 图像分割（灰度小于阈值为液滴）
mask = (gray < best_thresh).astype(np.uint8) * 255
# 反转 mask：白→黑，黑→白
inv = cv2.bitwise_not(mask)

# 找到所有连通域
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv)

# 创建一个空白图，用于存放“洞”
holes = np.zeros_like(mask)

# 遍历所有连通域，保留那些被白色完全包围的区域
for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    # 如果连通域不接触边界，则它是一个“洞”
    if x > 0 and y > 0 and x + w < mask.shape[1] and y + h < mask.shape[0]:
        holes[labels == i] = 255

# 将洞填回白色区域
mask_filled = cv2.bitwise_or(mask, holes)


# Step 7: 可视化
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Step A: 连通域分析（只保留两个最大连通域）
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_filled)

areas = stats[1:, cv2.CC_STAT_AREA]
sorted_idx = np.argsort(areas)[::-1]
keep_labels = sorted_idx[:2] + 1

# Step B: 对每个连通域做最小外接圆（不会包含拖尾）
circles = []
for lbl in keep_labels:
    ys, xs = np.where(labels == lbl)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cx, cy), r = cv2.minEnclosingCircle(pts)
    circles.append((cx, cy, r))

# Step C: 可视化
vis = cv2.cvtColor(mask_filled, cv2.COLOR_GRAY2BGR)
for (cx, cy, r) in circles:
    cv2.circle(vis, (int(cx), int(cy)), int(r), (0,0,255), 2)

plt.figure(figsize=(8,6))
plt.imshow(vis)
plt.title("Droplets with Enclosing Circles (Tail Removed)")


# 原始灰度图
axs[0, 0].imshow(gray, cmap='gray')
axs[0, 0].set_title("Gray Image")
axs[0, 0].axis('off')

# 灰度直方图 + 标注峰和谷
axs[0, 1].plot(hist, color='black')
axs[0, 1].plot(peaks, hist[peaks], "ro", label="Peaks")
axs[0, 1].plot(valleys, hist[valleys], "bo", label="Valleys")
axs[0, 1].axvline(best_thresh, color='green', linestyle='--', label=f"Best Threshold = {best_thresh}")
axs[0, 1].set_title("Gray Histogram")
axs[0, 1].set_xlabel("Gray Level")
axs[0, 1].set_ylabel("Pixel Count")
axs[0, 1].legend()

# 分割掩膜
axs[1, 0].imshow(mask_filled, cmap='gray')
axs[1, 0].set_title("Segmentation Mask")
axs[1, 0].axis('off')

# 原图叠加分割结果
overlay = cv2.addWeighted(gray, 0.7, mask_filled, 0.3, 0)
axs[1, 1].imshow(overlay, cmap='gray')
axs[1, 1].set_title("Overlay Result")
axs[1, 1].axis('off')

plt.tight_layout()
plt.show()
