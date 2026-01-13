import cv2
import numpy as np

def preprocess(frame, use_clahe=False):
    """
    对输入图像预处理，输出光照均衡增强后的灰度图

    参数:
        frame (numpy.ndarray): 输入的彩色图像

    返回:
        img_proc:预处理后的灰度图像
    """
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 去除噪声
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用CLAHE进行光照均衡增强
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_proc = clahe.apply(blur)
    else:
        img_proc = blur

    return img_proc