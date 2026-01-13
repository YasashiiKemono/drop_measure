import cv2
import numpy as np
import itertools


# ============================
# 抗锯齿预处理模块
# ============================
def preprocess_antialias(frame):
    """
    对原始图像进行抗锯齿处理：
    - 上采样（2倍）
    - 双边滤波（保边去噪）
    - 返回灰度图
    """
    # 上采样
    up = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 双边滤波（保边去噪）
    smooth = cv2.bilateralFilter(up, d=9, sigmaColor=75, sigmaSpace=75)

    # 灰度化
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)

    return gray


# ============================
# 圆质量评分模块
# ============================
def score_circles(circles, edges_closed):
    """
    对 Hough 检测到的圆进行评分：
    - 数量越多越好（但不超过阈值）
    - 半径一致性越高越好（低方差）
    - 圆边缘强度越高越好（在 Canny 边缘图上采样）

    返回：
        score: float，越高越好
    """
    if circles is None or len(circles[0]) == 0:
        return 0

    circles = np.uint16(np.around(circles))
    radii = [c[2] for c in circles[0]]
    num = len(radii)

    # 半径一致性评分（方差越小越好）
    radius_std = np.std(radii)
    radius_score = 1 / (1 + radius_std)

    # 边缘强度评分（在圆周上采样）
    edge_score = 0
    for c in circles[0]:
        x, y, r = c
        count = 0
        for theta in range(0, 360, 10):
            dx = int(x + r * np.cos(np.deg2rad(theta)))
            dy = int(y + r * np.sin(np.deg2rad(theta)))
            if 0 <= dx < edges_closed.shape[1] and 0 <= dy < edges_closed.shape[0]:
                count += edges_closed[dy, dx] > 0
        edge_score += count / 36  # 每个圆最多36个点

    edge_score /= num  # 平均边缘强度

    # 综合评分
    score = num * radius_score * edge_score
    return score


# ============================
# 自动调参主函数
# ============================
def auto_hough_tune(img):
    """
    自动搜索最佳 Hough 参数组合：
    - 遍历 param2, minDist, minRadius/maxRadius
    - 对每组参数进行圆检测 + 评分
    - 返回最佳参数和对应圆
    """

    # Step 1: 抗锯齿预处理
    img_proc = preprocess_antialias(img)

    # Step 2: Canny + 边缘闭合
    edges = cv2.Canny(img_proc, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 3: 参数搜索空间
    param2_list = [15, 20, 25, 30]
    minDist_list = [15, 20, 25]
    radius_range = [(10, 30), (15, 35), (20, 40)]

    best_score = -1
    best_circles = None
    best_params = None

    for p2, md, (rmin, rmax) in itertools.product(param2_list, minDist_list, radius_range):
        circles = cv2.HoughCircles(
            edges_closed,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=md,
            param1=100,
            param2=p2,
            minRadius=rmin,
            maxRadius=rmax
        )

        score = score_circles(circles, edges_closed)
        if score > best_score:
            best_score = score
            best_circles = circles
            best_params = (p2, md, rmin, rmax)

    return best_circles, best_params, edges_closed


# ============================
# 可视化测试入口
# ============================
if __name__ == "__main__":
    img = cv2.imread("frame.png")
    if img is None:
        print("无法读取图像")
        exit()

    circles, params, edges = auto_hough_tune(img)

    vis = cv2.cvtColor(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0]:
            cv2.circle(vis, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(vis, (c[0], c[1]), 2, (0, 0, 255), 3)

    print(f"最佳参数: param2={params[0]}, minDist={params[1]}, minRadius={params[2]}, maxRadius={params[3]}")
    cv2.imshow("edges", edges)
    cv2.imshow("best circles", vis)
    cv2.waitKey(0)
