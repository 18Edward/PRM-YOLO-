import os
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = r"E:\tcy_data\nn\YOLOv11\ultralytics-main\runs\detect\train16\weights\best.pt"  # 替换为你的模型路径
IMAGE_FOLDER = r"C:\Users\wyk31\Desktop\nn论文\test_pic"                   # 替换为你的图片文件夹路径
OUTPUT_DIR = "output_folder_based"                    # 推理结果保存目录

# 支持的图像扩展名
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

model = YOLO(MODEL_PATH)

image_folder = Path(IMAGE_FOLDER)
assert image_folder.is_dir(), f"Image folder {IMAGE_FOLDER} does not exist!"

image_paths = [p for p in image_folder.rglob('*') if p.suffix.lower() in IMG_EXTENSIONS]
print(f"Found {len(image_paths)} images in {IMAGE_FOLDER}")

if not image_paths:
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = model.predict(
    source=image_paths,
    save=True,               # 保存带检测框的图像
    save_txt=False,          # 如需保存 txt 标注可设为 True
    project=OUTPUT_DIR,      # 保存根目录
    name="detected",         # 结果子目录名: output_infer/detected/
    exist_ok=True,           # 允许覆盖同名目录
    conf=0.25,               # 置信度阈值（根据你的训练调整）
    iou=0.65,                # NMS IOU 阈值
    device='',               # 自动选择 GPU/CPU；可指定 'cuda:0' 或 'cpu'
    verbose=True             # 打印进度
)
print(f"\n✅Inference completed! Results saved to: {OUTPUT_DIR}/detected/")