import cv2
import numpy as np


def segment_liquid_droplets(frame_bgr):
    """
    使用颜色定位液滴区域，在灰度图中重建边缘并拟合圆。
    返回拟合结果图像和液滴信息列表。
    """

    # Step 1: 转 HSV 空间
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Step 2: 精准粉红色范围（反推自你图像）
    lower_pink = np.array([155, 80, 80])
    upper_pink = np.array([175, 255, 255])

    # Step 3: 颜色分割 → 得到液滴区域 mask
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Step 4: 扩张 + 闭运算 → 包含黑色边缘 + 连通碎块
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Step 5: 灰度图中提取液滴区域
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Step 6: Canny 边缘检测（只在液滴区域内）
    edges = cv2.Canny(masked_gray, 50, 150)

    # Step 7: 边缘闭运算 → 修复断裂
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 8: 提取轮廓
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    droplets = []
    result_img = frame_bgr.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # 过滤小噪点
            continue

        # 拟合最小外接圆
        (x, y), r = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(r)

        # 画圆
        cv2.circle(result_img, center, radius, (0, 255, 0), 2)
        cv2.circle(result_img, center, 2, (0, 0, 255), -1)

        droplets.append((x, y, r))

    return result_img, edges_closed, droplets


# ============================
# 自测入口
# ============================
if __name__ == "__main__":
    img = cv2.imread("data/test.png")  # 替换为你的图像路径
    if img is None:
        print("无法读取图像")
        exit()

    result, edge_map, droplets = segment_liquid_droplets(img)

    print(f"检测到 {len(droplets)} 个液滴")
    for i, (x, y, r) in enumerate(droplets):
        print(f"液滴 {i+1}: 圆心=({x:.1f}, {y:.1f}), 半径={r:.1f}px, 直径={2*r:.1f}px")

    cv2.imshow("原图", img)
    cv2.imshow("边缘图", edge_map)
    cv2.imshow("拟合结果", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
