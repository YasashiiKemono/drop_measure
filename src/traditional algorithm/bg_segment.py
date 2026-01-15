import cv2
import numpy as np
import math


# ============================
# 参数区（你可以按需要调整）
# ============================
VIDEO_PATH = "video1.wmv"  # 输入视频路径
N_BG_FRAMES = 50          # 用多少帧来估计背景
FRAME_STEP = 5            # 每隔多少帧取一帧做背景
DIFF_THRESH = 20          # 背景差分阈值（越小越敏感）
MIN_DROPLET_AREA = 200    # 液滴最小面积过滤
ANNULUS_INNER_SCALE = 1.12
ANNULUS_OUTER_SCALE = 1.20


# ============================
# 工具函数
# ============================
def apply_CLAHE(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


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


def estimate_background(video_path, n_frames=50, step=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    idx = 0
    collected = 0
    while collected < n_frames and idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.astype(np.float32))
        collected += 1
        idx += step

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("无法收集到用于背景建模的帧")

    stack = np.stack(frames, axis=0)  # (N, H, W)
    background = np.median(stack, axis=0).astype(np.uint8)
    return background


def get_droplet_mask_from_background(frame_gray, background_gray):
    diff = cv2.absdiff(frame_gray, background_gray)
    # 背景差分阈值
    _, fg = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)

    # 形态学清理
    kernel = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 可选：限制在中间通道区域（根据你实际图像可加 ROI 裁剪）
    return fg


def fit_droplets_and_draw(frame_bgr, frame_gray, droplet_mask):
    result = frame_bgr.copy()
    contours, _ = cv2.findContours(droplet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_DROPLET_AREA:
            continue

        # 1. 用 droplet_mask 拟合粗圆 ROI1
        (x, y), r = cv2.minEnclosingCircle(cnt)
        cx0, cy0, r0 = int(x), int(y), int(r)

        
        # 4. 可视化：粗圆 + 环带 + 拟合圆
        cv2.circle(result, (cx0, cy0), r0, (0, 255, 255), 2)        # 粗圆（黄）
        

    return result


# ============================
# 主流程
# ============================
if __name__ == "__main__":
    # 1. 背景建模
    print("估计背景中...")
    background = estimate_background(VIDEO_PATH, n_frames=N_BG_FRAMES, step=FRAME_STEP)
    cv2.imshow("background", background)
    cv2.waitKey(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("无法打开视频")
        exit()

    # 你可以只处理某一帧，用于调试
    target_frame_idx = 0  # 例如第 0 帧，之后可以改成循环处理
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("无法读取目标帧")
        exit()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. 背景差分 → 反向抠出液滴区域
    droplet_mask = get_droplet_mask_from_background(gray, background)
    cv2.imshow("droplet_mask", droplet_mask)
    cv2.waitKey(1)

    # 3. 在当前帧上做 EDBR‑Annulus + RANSAC 拟合
    result = fit_droplets_and_draw(frame, gray, droplet_mask)

    cv2.imshow("result", result)
    cv2.imwrite("result.png", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()