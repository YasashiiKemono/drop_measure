
import cv2
import numpy as np
import random
import math


def ransac_circle(points, n_iter=2000, inlier_thresh=2.0, min_inliers_ratio=0.3):
    """
    在给定点集上做 RANSAC 圆拟合。
    points: (N, 2)，每行 (x, y)
    返回: (cx, cy, r, inliers_mask) 或 None
    """

    if len(points) < 10:
        return None

    pts = points.astype(np.float32)
    best_model = None
    best_inliers = 0
    best_mask = None

    # 预先打乱索引避免每次 random.choices 成本
    idx = np.arange(len(pts))

    for _ in range(n_iter):
        # 随机选3个点拟合圆
        sample_idx = np.random.choice(idx, size=3, replace=False)
        p1, p2, p3 = pts[sample_idx]

        # 三点共线就跳过
        denom = 2 * (p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
        if abs(denom) < 1e-6:
            continue

        # 三点确定圆心（解析几何公式）
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        a = x1**2 + y1**2
        b = x2**2 + y2**2
        c = x3**2 + y3**2

        cx = (a*(y2 - y3) + b*(y3 - y1) + c*(y1 - y2)) / denom
        cy = (a*(x3 - x2) + b*(x1 - x3) + c*(x2 - x1)) / denom

        # 半径
        dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        r = np.median(dists)  # 用中位数更鲁棒

        # 计算 inliers
        residuals = np.abs(dists - r)
        inliers_mask = residuals < inlier_thresh
        n_inliers = np.count_nonzero(inliers_mask)

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_model = (cx, cy, r)
            best_mask = inliers_mask

    if best_model is None:
        return None

    # 检查 inlier 比例
    if best_inliers < min_inliers_ratio * len(pts):
        return None

    cx, cy, r = best_model

    # 用 inliers 再做一次精修（最小二乘拟合）
    inlier_pts = pts[best_mask]
    if len(inlier_pts) >= 10:
        cx, cy, r = refine_circle_least_squares(inlier_pts, (cx, cy, r))

    return cx, cy, r, best_mask


def refine_circle_least_squares(points, initial_model):
    """
    用最小二乘在 RANSAC 初值附近精修圆参数
    """
    cx, cy, r = initial_model
    x = points[:, 0]
    y = points[:, 1]

    # 线性化：x^2 + y^2 + Ax + By + C = 0
    # 中心和半径可从 A, B, C 求出
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    A_, B_, C_ = params
    cx_ls = -A_ / 2
    cy_ls = -B_ / 2
    r_ls = math.sqrt(max(cx_ls**2 + cy_ls**2 - C_, 0))

    return cx_ls, cy_ls, r_ls


def detect_dark_edge_points(gray, pink_mask=None,
                            dark_thresh=50,
                            erode_iter=0):
    """
    检测“暗像素点”作为黑边候选。
    gray: 灰度图
    pink_mask: 液滴大致区域的 mask（可选，用于屏蔽背景）
    dark_thresh: 小于该值认为是“暗”
    erode_iter: 对暗点做一点腐蚀，去掉孤立噪声
    返回: (N, 2) 点坐标 (x, y)
    """
    if pink_mask is not None:
        # 先限制在液滴区域附近
        mask_region = pink_mask > 0
    else:
        mask_region = np.ones_like(gray, dtype=bool)

    dark = (gray < dark_thresh) & mask_region

    if erode_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        dark = cv2.erode(dark.astype(np.uint8)*255, kernel, iterations=erode_iter) > 0

    ys, xs = np.where(dark)
    points = np.column_stack([xs, ys])
    return points


def rough_pink_mask(frame_bgr):
    """
    粗略用 HSV 把粉色区域圈出来，只用来限制搜索范围，不用于形状。
    参数可以之后再调。
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([155, 50, 50])
    upper_pink = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # 稍微膨胀一下，让 mask 覆盖黑边
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


if __name__ == "__main__":
    # 1. 读取图像
    img = cv2.imread("data/Image1.jpg")  # 换成你当前那帧的文件名
    if img is None:
        print("无法读取图像")
        exit()

    # 2. 灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. 粗略粉色区域 mask（只用来缩小范围）
    pink_mask = rough_pink_mask(img)

    # 4. 检测暗边候选点
    dark_points = detect_dark_edge_points(
        gray,
        pink_mask=pink_mask,
        dark_thresh=50,   # 这里是关键参数之一
        erode_iter=0
    )
    print(f"检测到暗点数量: {len(dark_points)}")

    vis = img.copy()
    for x, y in dark_points:
        cv2.circle(vis, (int(x), int(y)), 1, (0, 0, 255), -1)

    # 5. 用 RANSAC 拟合圆（先假定图中只有一个主要液滴）
    model = ransac_circle(
        dark_points,
        n_iter=2000,
        inlier_thresh=2.0,      # 点到圆的距离误差容忍
        min_inliers_ratio=0.2   # 至少有 20% 点在圆上
    )

    if model is None:
        print("RANSAC 未找到稳定圆模型")
    else:
        cx, cy, r, inlier_mask = model
        print(f"拟合圆: center=({cx:.1f}, {cy:.1f}), r={r:.1f}px, d={2*r:.1f}px")
        # 画 inliers
        inlier_pts = dark_points[inlier_mask]
        for x, y in inlier_pts:
            cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0), -1)

        # 画拟合圆
        cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 255), 2)
        cv2.circle(vis, (int(cx), int(cy)), 2, (255, 0, 0), -1)

    # # 6. 可视化
    # cv2.imshow("gray", gray)
    # cv2.imshow("rough_pink_mask", pink_mask)
    # cv2.imshow("dark_points + fitted circle", vis)
    # cv2.imwrite("rough_pink_mask.png", pink_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print("灰度图统计:")
    print("min:", np.min(gray))
    print("max:", np.max(gray))
    print("mean:", np.mean(gray))
    print("std:", np.std(gray))
