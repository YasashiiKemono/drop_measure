from labelme import utils
import numpy as np
import json
import os

json_file = "Image1.json"

# 读取 JSON
data = json.load(open(json_file))

# 自动生成 label 映射
label_name_to_value = {"_background_": 0}
for shape in data["shapes"]:
    label = shape["label"]
    if label not in label_name_to_value:
        label_name_to_value[label] = len(label_name_to_value)

# 生成 labelme 的整数 mask（0,1,2,...）
lbl, _ = utils.shapes_to_label(
    img_shape=(data["imageHeight"], data["imageWidth"], 3),
    shapes=data["shapes"],
    label_name_to_value=label_name_to_value,
)

# 转成 0/255 的二值 mask，并确保 dtype=uint8
binary_mask = (lbl > 0).astype(np.uint8) * 255

# ⭐ 关键：不再用 labelme 的 lblsave，而是用 cv2.imwrite
import cv2
out_name = json_file.replace(".json", "_mask.png")
cv2.imwrite(out_name, binary_mask)

print(f"Saved: {out_name}")
