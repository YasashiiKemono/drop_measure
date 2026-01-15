import cv2
import numpy as np


# =======================
# 工具函数：填补二值图中的“洞”
# =======================
def fill_holes(binary: np.ndarray) -> np.ndarray:
    """
    对二值图像进行“填洞”操作：
    - 输入 binary: uint8, 0/255，液滴=255，背景=0
    - 输出 filled: uint8, 0/255，液滴内部的小黑洞会被填成白色

    原理：
    1. 用 floodFill 从背景（左上角）开始泛洪，把背景全部填成白色
    2. 取反得到只有“洞”的图
    3. 和原二值图做 OR，把洞补回去
    """
    h, w = binary.shape[:2]

    # 复制一份作为 floodFill 的输入
    flood = binary.copy()

    # floodFill 的 mask 必须比原图大 2 像素
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # 从 (0, 0) 作为背景种子点开始泛洪，新的填充值为 255（白）
    cv2.floodFill(flood, mask, seedPoint=(0, 0), newVal=255)

    # 现在 flood 中：背景区域 = 255，洞 = 0，液滴区域原样保留
    # 取反后：背景 = 0，洞 = 255，液滴边缘附近也保留一圈
    flood_inv = cv2.bitwise_not(flood)

    # 原二值图 OR 上“洞图”，把洞填回去
    filled = binary | flood_inv

    return filled


# =======================================
# 工具函数：可选的分水岭分割（用于轻微粘连）
# =======================================
def watershed_separation(binary: np.ndarray) -> np.ndarray:
    """
    对二值图像做简单的分水岭分割，用于分离轻微粘连的液滴。

    输入：
        binary: uint8, 0/255，液滴=255，背景=0
    输出：
        markers: int32，每个液滴一个 label，背景为 1，分水岭边界为 -1

    如果你的液滴基本不粘连，可以在主函数里把这个步骤关掉。
    """
    bin_u8 = binary.astype(np.uint8)

    # 小形态学开运算，去掉非常小的噪声点
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(bin_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    # 膨胀得到“确定背景区域”
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 距离变换，液滴中心会有较大值，用它来找“确定前景”
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # 阈值分割距离图：取大于 0.4 * max 的区域作为确定前景
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 未知区域 = 背景 - 前景
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 对确定前景做连通域标记，得到初始 markers
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # OpenCV 分水岭要求：背景 > 0，所以把所有 label + 1
    markers = markers + 1

    # 未知区域标记为 0，让分水岭去决定
    markers[unknown == 255] = 0

    # 分水岭需要一个三通道图像作为“地形图”
    img_color = cv2.cvtColor(bin_u8, cv2.COLOR_GRAY2BGR)

    # 真正的分水岭算法
    markers = cv2.watershed(img_color, markers)

    # 返回的 markers:
    #   -1：分水岭边界
    #    1：背景
    #   >=2：各个液滴的标签
    return markers


# =======================
# 主函数：分割液滴
# =======================
def segment(
    img_proc: np.ndarray,
    thr: int = 120,
    use_watershed: bool = False
):
    """
    对预处理后的灰度图进行分割。

    输入：
        img_proc: 预处理后的灰度图，uint8，0~255
        thr: 固定阈值（液滴暗，背景亮） —— 可根据直方图手动调整
        use_watershed: 是否启用分水岭分割（一般在液滴有轻微粘连时使用）

    输出：
        binary: 填洞 + 形态学处理后的二值图 (uint8, 0/255)
        edges: Canny 边缘图 (uint8, 0/255)
        markers: 
            - 如果 use_watershed=True：返回分水岭标签图 (int32)
            - 如果 use_watershed=False：返回 None

    推荐使用方式：
        1) 先调用 segment() 得到 binary, edges, markers
        2) 再在 contour.py 中基于 binary 或 markers 提取轮廓
    """

    # ---------- Step 1: 固定阈值 ----------
    # 液滴比背景暗，所以用 THRESH_BINARY_INV：
    #   灰度 < thr 的液滴区域 → 变成 255（白）
    #   灰度 >= thr 的背景区域 → 变成 0（黑）
    #
    # thr 的选择建议：
    #   - 先画直方图，看液滴峰和背景峰
    #   - 选在“中间谷值附近”（比如液滴在 40~50，背景在 190~200，可以选 120 左右）
    _, binary = cv2.threshold(
        img_proc,
        thr,
        255,
        cv2.THRESH_BINARY_INV
    )

    # ---------- Step 2: 填洞 ----------
    # 如果液滴内部有亮斑（灰度高于 thr），阈值后会出现黑洞。
    # 填洞后，液滴区域会变成一个实心的白色区域，便于后续轮廓提取。
    binary = fill_holes(binary)

    # ---------- Step 3: 形态学闭运算 ----------
    # 作用：
    #   - 闭合边缘上的小缺口（尤其是锯齿边缘中间的小裂缝）
    #   - 连接非常窄的小断裂，让轮廓更加连续、光滑
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ---------- Step 4: Canny 边缘（用于可视化/调试） ----------
    # Canny 直接作用在 binary 上，这样边缘非常干净：
    #   - 不受背景纹理影响
    #   - 只在液滴轮廓附近有响应
    edges = cv2.Canny(binary, 50, 150)

    # ---------- Step 5: 可选的分水岭 ----------
    # 如果液滴基本不粘连，可以关闭分水岭：
    #   use_watershed = False
    # 如果你发现有些液滴轻微粘在一起，可以打开：
    #   use_watershed = True
    if use_watershed:
        markers = watershed_separation(binary)
    else:
        markers = None

    return binary, edges, markers


# =======================
# 简单自测入口（可选）
# =======================
if __name__ == "__main__":
    """
    这是一个简单的自测代码：
    - 假设你有一张预处理后的灰度图 "img_proc_debug.png"
    - 直接运行 python segment.py 就能看到分割效果
    """
    img_proc = cv2.imread("img_proc_debug.png", cv2.IMREAD_GRAYSCALE)
    if img_proc is None:
        print("无法读取 img_proc_debug.png，请检查路径。")
        exit(0)

    binary, edges, markers = segment(img_proc, thr=120, use_watershed=False)

    cv2.imshow("img_proc", img_proc)
    cv2.imshow("binary", binary)
    cv2.imshow("edges", edges)

    if markers is not None:
        # 简单可视化分水岭标签：把 int32 映射到伪彩色
        markers_vis = cv2.convertScaleAbs(markers, alpha=10)
        markers_vis = cv2.applyColorMap(markers_vis, cv2.COLORMAP_JET)
        cv2.imshow("markers", markers_vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
