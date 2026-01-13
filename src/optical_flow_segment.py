import cv2
import numpy as np


# ============================
# 辅助函数：去除小连通域
# ============================
def remove_small_components(mask, min_area=200):
    """
    去掉面积太小的连通区域，保留真正的液滴区域。

    输入：
        mask: uint8 二值图，0/255
        min_area: 最小保留面积（像素）

    输出：
        cleaned: uint8 二值图，0/255
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):  # 0 是背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


# ============================
# 核心类：增强版光流分割器
# ============================
class OpticalFlowSegmenter:
    def __init__(self,
                 pyr_scale=0.5,
                 levels=3,
                 winsize=15,
                 iterations=3,
                 poly_n=5,
                 poly_sigma=1.2,
                 mag_alpha=0.8,
                 base_thresh=0.5,
                 min_area=200):
        """
        初始化光流分割器。

        参数说明（基本不用频繁改）：
            pyr_scale: 图像金字塔缩放比例
            levels: 金字塔层数
            winsize: 每一层窗口大小
            iterations: 每层迭代次数
            poly_n, poly_sigma: 多项式展开参数（平滑程度）
            mag_alpha: 时间平滑系数，越大越平滑（0~1）
            base_thresh: 基础阈值偏移，用于区分背景/液滴（单位：像素/帧）
            min_area: 最小连通域面积，用于去噪
        """
        self.prev_gray = None          # 前一帧灰度图
        self.mag_smooth = None         # 时间平滑后的光流幅度
        self.mag_alpha = mag_alpha     # 时间平滑权重
        self.base_thresh = base_thresh
        self.min_area = min_area

        # Farneback 参数
        self.fb_params = dict(
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0
        )

    def reset(self):
        """重置内部状态，用在切换视频或重开始时。"""
        self.prev_gray = None
        self.mag_smooth = None

    def process(self, frame_bgr):
        """
        处理一帧，返回液滴 mask。

        输入：
            frame_bgr: 当前帧，BGR 彩色图

        输出：
            mask: uint8 二值图，0/255，255 为“运动区域”（液滴）
            mag_vis: 可视化用的光流幅度图（归一化到 0~255）
        """
        # 转灰度
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 第一帧：无法计算光流，返回空 mask
        if self.prev_gray is None:
            self.prev_gray = gray
            h, w = gray.shape
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            empty_vis = np.zeros((h, w), dtype=np.uint8)
            return empty_mask, empty_vis

        # 计算 Farneback 光流：prev_gray -> gray
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            **self.fb_params
        )

        # 分解光流到 x/y 分量
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        # 光流幅度（每个像素的速度大小）
        mag = np.sqrt(flow_x**2 + flow_y**2)

        # 空间平滑（消除局部噪点）
        mag_blur = cv2.GaussianBlur(mag, (7, 7), 0)

        # 时间平滑（消除随机闪烁噪声，强化持续运动）
        if self.mag_smooth is None:
            self.mag_smooth = mag_blur
        else:
            self.mag_smooth = self.mag_alpha * self.mag_smooth + (1 - self.mag_alpha) * mag_blur

        # 背景速度建模：背景大部分应该是低速
        # 用中位数估计“背景速度”
        bg_speed = np.median(self.mag_smooth)

        # 自适应阈值：背景均值 + 偏移
        # 偏移可以理解为“至少比背景快多少才算液滴”
        thresh_val = bg_speed + self.base_thresh

        # 生成初始 mask
        mask = (self.mag_smooth > thresh_val).astype(np.uint8) * 255

        # 形态学开运算：去掉孤立噪点
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 形态学闭运算：填补小裂缝
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 去除小连通域噪声
        if self.min_area is not None and self.min_area > 0:
            mask = remove_small_components(mask, min_area=self.min_area)

        # 生成可视化用的 magnitude 图（0~255）
        mag_vis = cv2.normalize(self.mag_smooth, None, 0, 255, cv2.NORM_MINMAX)
        mag_vis = mag_vis.astype(np.uint8)

        # 更新 prev_gray
        self.prev_gray = gray

        return mask, mag_vis


# ============================
# 自测 / 示例：处理一段视频
# ============================
if __name__ == "__main__":
    # 打开视频（可以是 mp4，也可以是摄像头）
    cap = cv2.VideoCapture("video1.wmv")
    # cap = cv2.VideoCapture(0)  # 如果要直接用摄像头

    seg = OpticalFlowSegmenter(
        mag_alpha=0.8,    # 时间平滑力度
        base_thresh=0.5,  # 相对背景的速度偏移阈值
        min_area=200      # 最小液滴面积（根据分辨率调）
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask, mag_vis = seg.process(frame)

        # 可视化：把 mask 叠加到原图上
        overlay = frame.copy()
        overlay[mask == 255] = [0, 0, 255]  # 把液滴区域标成红色

        vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        cv2.imshow("frame", frame)
        cv2.imshow("flow_magnitude", mag_vis)
        cv2.imshow("mask", mask)
        cv2.imshow("overlay", vis)

        key = cv2.waitKey(1)
        if key == 27:  # ESC 退出
            break

    cap.release()
    cv2.destroyAllWindows()
