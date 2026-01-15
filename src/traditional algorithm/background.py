# 背景建模模块
import cv2
import numpy as np

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
        raise RuntimeError("无法收集背景帧")

    background = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    return background
