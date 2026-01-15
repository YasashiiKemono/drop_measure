import cv2
import numpy as np

MIN_DROPLET_AREA = 200

ROI_X1, ROI_Y1 = 540, 390
ROI_X2, ROI_Y2 = 680, 500

def fit_circle_least_squares(points):
    """
    最小二乘圆拟合（高速液滴最稳定）
    points: Nx2 numpy 数组
    返回: (cx, cy, r)
    """
    x = points[:, 0]
    y = points[:, 1]

    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x*x + y*y

    c, d, e = np.linalg.lstsq(A, b, rcond=None)[0]

    cx = c
    cy = d
    r = np.sqrt(e + cx*cx + cy*cy)

    return cx, cy, r


def detect_droplets(mask):
    mask_roi = mask[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]

    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_DROPLET_AREA:
            continue

        pts = cnt.reshape(-1, 2).astype(np.float32)

        # 最小二乘圆拟合（核心）
        cx, cy, r = fit_circle_least_squares(pts)

        # ROI → 全图坐标
        cx += ROI_X1
        cy += ROI_Y1

        circles.append((int(cx), int(cy), int(r)))

    return circles
