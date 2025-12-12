import os
from pathlib import Path
from ultralytics import YOLO
import shutil

# ---------- 配置 ----------
MODEL_PATH = r"E:\tcy_data\nn\YOLOv11\ultralytics-main\runs\detect\train104\weights\best.pt"
IMAGE_FOLDER = r"C:\Users\wyk31\Desktop\nn论文\test_pic"
OUTPUT_DIR = "output_folder_v8based_swapped"

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# ---------- 加载模型 ----------
model = YOLO(MODEL_PATH)

# ---------- 构建对调后的类别名 ----------
original_names = model.names.copy()  # {0: 'person', 1: 'car', ...}
print("原始类别:", original_names)

# 找到 person 和 car 的索引
person_idx = None
car_idx = None
for idx, name in original_names.items():
    if name == 'person':
        person_idx = idx
    elif name == 'car':
        car_idx = idx

if person_idx is not None and car_idx is not None:
    # 创建新的 names 字典（仅用于显示）
    swapped_names = original_names.copy()
    swapped_names[person_idx] = 'car'
    swapped_names[car_idx] = 'person'
    print(f"✅ 将对调: person({person_idx}) ↔ car({car_idx})")
else:
    print("⚠️ 未同时找到 'person' 和 'car'，跳过对调")
    swapped_names = original_names

# ---------- 处理图像 ----------
image_folder = Path(IMAGE_FOLDER)
assert image_folder.is_dir(), f"Image folder {IMAGE_FOLDER} does not exist!"

image_paths = [p for p in image_folder.rglob('*') if p.suffix.lower() in IMG_EXTENSIONS]
print(f"Found {len(image_paths)} images")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 临时保存原始 names（可选）
original_model_names = model.names

# 关键：临时替换 model.names 为 swapped_names
model.names = swapped_names

# 推理并保存（使用官方 plot）
for img_path in image_paths:
    results = model(img_path, conf=0.25, iou=0.65, device='', verbose=False)
    result = results[0]

    # 使用官方 plot 方法（效果 100% 一致！）
    plotted_img = result.plot(
        labels=True,
        boxes=True,
        conf=True,
        line_width=None,      # 自动根据图像大小调整（默认行为）
        font_size=None,       # 同上
        font='Arial.ttf',     # 默认字体
        pil=True              # 返回 PIL Image
    )

    # 保存
    output_path = os.path.join(OUTPUT_DIR, img_path.name)
    plotted_img.save(output_path)
    print(f"Saved: {output_path}")

# （可选）恢复原始 names
model.names = original_model_names

print(f"\n✅ 推理完成！结果保存至: {OUTPUT_DIR}")