import cv2
import matplotlib.pyplot as plt

# 读取图像（BGR）
img = cv2.imread("data/Image1.jpg")

# 转灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 显示灰度图
plt.imshow(gray, cmap='gray')
plt.title("Gray Image")
plt.axis('off')
plt.show()

# 查看灰度直方图
plt.figure()
plt.hist(gray.ravel(), bins=256, range=(0, 256))
plt.title("Gray Histogram")
plt.xlabel("Gray Level")
plt.ylabel("Pixel Count")
plt.show()
