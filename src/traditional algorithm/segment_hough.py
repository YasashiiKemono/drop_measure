import cv2
import numpy as np


def segment_hough(img_proc):
    """
    使用 Canny + HoughCircles 对液滴进行检测。
    适用于：
        - USB 摄像头拍摄的视频
        - 灰度分布混乱（无法阈值分割）
        - 液滴边缘清晰但亮度不稳定
        - 分辨率不高、边缘有锯齿

    输入：
        img_proc: 预处理后的灰度图（uint8, 0~255）

    输出：
        edges_closed: 经过闭运算的边缘图（用于调试）
        circles: Hough 检测到的圆（None 或 Nx3 数组）
    """

    # ============================
    # Step 1: Canny 边缘检测
    # ============================
    # 说明：
    #   - Canny 只看梯度，不看亮度
    #   - 即使亮度被摄像头自动曝光搞乱，边缘仍然稳定
    #   - 这是你当前图像最可靠的特征
    edges = cv2.Canny(img_proc, 50, 150)

    # ============================
    # Step 2: 边缘闭运算（修复锯齿 + 断裂）
    # ============================
    # 说明：
    #   - USB 摄像头分辨率低，液滴边缘呈锯齿状
    #   - Canny 会产生断裂边缘
    #   - 闭运算能把断裂边缘补上，让 Hough 更容易检测
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ============================
    # Step 3: Hough 圆检测
    # ============================
    # 说明：
    #   - dp：累积分辨率（越大越快，但精度略降）
    #   - minDist：圆之间最小距离（液滴密集 → 设小一点）
    #   - param1：Canny 高阈值（越高 → 边缘越干净）
    #   - param2：投票阈值（越低 → 检测更多圆，但噪声也多）
    #   - minRadius/maxRadius：液滴半径范围（根据视频调整）
    circles = cv2.HoughCircles(
        edges_closed,
        cv2.HOUGH_GRADIENT,
        dp=1.2,              # 越大越快，1.2 是一个很好的折中
        minDist=15,          # 液滴之间的最小距离（可调）
        param1=100,          # Canny 高阈值（可调）
        param2=215,           # 投票阈值（越低越容易检测到圆）
        minRadius=10,        # 液滴最小半径（可调）
        maxRadius=30         # 液滴最大半径（可调）
    )

    return edges_closed, circles


# ============================
# 自测入口（可选）
# ============================
if __name__ == "__main__":
    img = cv2.imread("img_proc_debug.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("无法读取测试图像")
        exit()

    edges, circles = segment_hough(img)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            cv2.circle(vis, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(vis, (c[0], c[1]), 2, (0, 0, 255), 3)

    cv2.imshow("img", img)
    cv2.imshow("edges", edges)
    cv2.imshow("circles", vis)
    cv2.waitKey(0)
