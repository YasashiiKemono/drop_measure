import cv2
import numpy as np
import math


# ============================
# 局部对比度增强（CLAHE）
# ============================
def apply_CLAHE(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ============================
# LoG + 零交叉检测
# ============================
def LoG_zero_crossing(gray, sigma=1.0, threshold=2.0):
    ksize = int(6 * sigma + 1)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    lap = cv2.Laplacian(blurred, cv2.CV_64F)

    edge = np.zeros_like(lap, dtype=np.uint8)

    for y in range(1, lap.shape[0] - 1):
        for x in range(1, lap.shape[1] - 1):
            patch = lap[y-1:y+2, x-1:x+2]
            p = lap[y, x]

            if (p > 0 and np.any(patch < 0)) or (p < 0 and np.any(patch > 0)):
                if np.max(np.abs(patch)) - np.min(np.abs(patch)) > threshold:
                    edge[y, x] = 255

    return edge


# ============================
# RANSAC 圆拟合
# ============================
def ransac_circle(points, n_iter=3000, inlier_thresh=3.0, min_inliers_ratio=0.1):
    if len(points) < 10:
        return None

    pts = points.astype(np.float32)
    best_model = None
    best_inliers = 0
    best_mask = None

    idx = np.arange(len(pts))

    for _ in range(n_iter):
        sample_idx = np.random.choice(idx, size=3, replace=False)
        p1, p2, p3 = pts[sample_idx]

        denom = 2 * (p1[0]*(p2[1]-p3[1]) +
                     p2[0]*(p3[1]-p1[1]) +
                     p3[0]*(p1[1]-p2[1]))
        if abs(denom) < 1e-6:
            continue

        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3

        a = x1**2 + y1**2
        b = x2**2 + y2**2
        c = x3**2 + y3**2

        cx = (a*(y2 - y3) + b*(y3 - y1) + c*(y1 - y2)) / denom
        cy = (a*(x3 - x2) + b*(x1 - x3) + c*(x2 - x1)) / denom

        dists = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
        r = np.median(dists)

        residuals = np.abs(dists - r)
        inliers_mask = residuals < inlier_thresh
        n_inliers = np.count_nonzero(inliers_mask)

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_model = (cx, cy, r)
            best_mask = inliers_mask

    if best_model is None:
        return None

    if best_inliers < min_inliers_ratio * len(pts):
        return None

    return best_model, best_mask


# ============================
# 最小二乘圆拟合
# ============================
def least_squares_circle(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x*x + y*y
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, d = c
    r = math.sqrt(cx*cx + cy*cy + d)
    return cx, cy, r


# ============================
# 主流程：径向投影 + EDBR‑Annulus
# ============================
if __name__ == "__main__":
    img = cv2.imread("data/Image1.jpg")  # 换成你当前那帧的文件名
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_pink = np.array([155, 50, 50])
    upper_pink = np.array([180, 255, 255])
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)

    kernel = np.ones((5, 5), np.uint8)
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        # 初始粗圆
        (x, y), r = cv2.minEnclosingCircle(cnt)
        cx0, cy0, r0 = int(x), int(y), int(r)

        print(f"\n=== 初始粗圆: center=({cx0},{cy0}), r0={r0} ===")

        # 阶段 1：宽环带（1.05~1.35）
        R_min = int(r0 * 1.05)
        R_max = int(r0 * 1.35)

        yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
        ring_mask = ((xx - cx0)**2 + (yy - cy0)**2 >= R_min**2) & \
                    ((xx - cx0)**2 + (yy - cy0)**2 <= R_max**2)

        clahe_img = apply_CLAHE(gray)
        ring_gray = np.zeros_like(gray)
        ring_gray[ring_mask] = clahe_img[ring_mask]

        edge = LoG_zero_crossing(ring_gray, sigma=1.0, threshold=2.0)
        edge = edge & ring_mask.astype(np.uint8) * 255

        ys, xs = np.where(edge > 0)
        pts = np.column_stack([xs, ys])

        print(f"宽环带边缘点数量: {len(pts)}")

        # 阶段 2：径向投影修正圆心
        angles = np.arctan2(pts[:, 1] - cy0, pts[:, 0] - cx0)
        radii = np.sqrt((pts[:, 0] - cx0)**2 + (pts[:, 1] - cy0)**2)

        bins = 720
        theta_bins = np.linspace(-np.pi, np.pi, bins+1)
        outer_points = []

        for i in range(bins):
            mask = (angles >= theta_bins[i]) & (angles < theta_bins[i+1])
            if np.sum(mask) < 1:
                continue
            idx = np.argmax(radii[mask])
            outer_points.append(pts[mask][idx])

        outer_points = np.array(outer_points)
        cx_fix, cy_fix, r_fix = least_squares_circle(outer_points)

        print(f"修正粗圆: center=({cx_fix:.2f},{cy_fix:.2f}), r={r_fix:.2f}")
        if len(outer_points) < 50:
            print("outer_points 太少，无法拟合圆")
            continue


        # 阶段 3：进入 EDBR‑Annulus（1.15~1.25）
        model = ransac_circle(outer_points)
        if model is None:
            print("RANSAC 拟合失败（outer_points 不足）")
            continue

        (model_params, mask) = model
        cx_final, cy_final, r_final = model_params

        if model is None:
            print("最终拟合失败")
            continue

        cx_final, cy_final, r_final = model
        cx_final, cy_final, r_final = int(cx_final), int(cy_final), int(r_final)

        print(f"最终圆: center=({cx_final},{cy_final}), r={r_final}")

        # 可视化
        cv2.circle(result, (cx0, cy0), r0, (0, 255, 255), 2)       # 初始粗圆（黄）
        cv2.circle(result, (int(cx_fix), int(cy_fix)), int(r_fix), (255, 0, 0), 2)  # 修正粗圆（蓝）
        cv2.circle(result, (cx_final, cy_final), r_final, (0, 255, 0), 2)           # 最终圆（绿）

    cv2.imshow("radial_projection_result", result)
    cv2.imwrite("radial_projection_result.png", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
