import cv2
import numpy as np
import matplotlib.pyplot as plt

from background import estimate_background
from mask import get_droplet_mask
from detector import detect_droplets


VIDEO_PATH = "video1.wmv"
OUTPUT_PATH = "output_final.mp4"

STABLE_THRESH = 25

# ROI（你的视频中稳定区域）
ROI_X1, ROI_Y1 = 540, 390
ROI_X2, ROI_Y2 = 680, 500


def is_mask_valid(mask):
    h, w = mask.shape
    total_area = h * w

    white_pixels = cv2.countNonZero(mask)
    if white_pixels == 0 or white_pixels > 0.15 * total_area:
        return False

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False

    max_cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_cnt) > 0.10 * total_area:
        return False

    return True


if __name__ == "__main__":
    print("Estimating background...")
    background = estimate_background(VIDEO_PATH, n_frames=50, step=5)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    # 计数逻辑
    last_seen = False
    count = 0

    # 直径（可选）
    last_valid_diameter = 0.0

    # 曲线数据（频率 = count / time）
    time_list = []
    freq_list = []
    diameter_list = []

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 稳定帧筛选
        diff = cv2.absdiff(gray, background)
        if diff.mean() >= STABLE_THRESH:
            out.write(result)
            frame_id += 1
            continue

        # mask
        mask = get_droplet_mask(gray, background)
        if not is_mask_valid(mask):
            out.write(result)
            frame_id += 1
            continue

        # ROI 内液滴检测
        circles = detect_droplets(mask)

        # 当前时间
        t = frame_id / fps

        # 是否有液滴在 ROI 内
        current_seen = len(circles) > 0

        if current_seen:
            cx, cy, r = circles[0]
            diameter_px = 2 * r

            # 画圆
            cv2.circle(result, (cx, cy), int(r), (0, 255, 255), 2)

            # ===== 计数：液滴进入 ROI =====
            if not last_seen:
                count += 1
                last_valid_diameter = diameter_px

                # 记录频率
                if t > 0:
                    freq = count / t
                    time_list.append(t)
                    freq_list.append(freq)
                    diameter_list.append(last_valid_diameter)

        last_seen = current_seen

        # 左上角显示
        # 避免 t = 0 时除零
        if t == 0:
            freq_display = 0
        else:
            freq_display = count / t

        text = f"Count: {count} | Freq: {freq_display:.2f} Hz | Dia: {last_valid_diameter:.1f} px"
        
        cv2.putText(result, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        out.write(result)
        frame_id += 1

    cap.release()
    out.release()
    print("Video processed:", OUTPUT_PATH)

    # 曲线绘制（频率）
    plt.figure(figsize=(10, 5))
    plt.plot(time_list, freq_list, label="Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Droplet Frequency vs Time")
    plt.grid(True)
    plt.legend()
    plt.savefig("frequency_curve.png", dpi=200)

    # 直径曲线（可选）
    plt.figure(figsize=(10, 5))
    plt.plot(time_list, diameter_list, label="Diameter (px)", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Diameter (px)")
    plt.title("Droplet Diameter vs Time")
    plt.grid(True)
    plt.legend()
    plt.savefig("diameter_curve.png", dpi=200)

    print("Curves saved: frequency_curve.png, diameter_curve.png")
