import cv2
import os

# ----------------------------
# 配置参数（请根据你的数据修改）
# ----------------------------
image_path = r"E:\tcy_data\datasets2\images\train\pic_29.jpg"
label_path = r"C:\Users\wyk31\Desktop\nn论文\0-1_change\pic_29.txt"  # 对应的 .txt 标注文件

# 类别名称列表（必须与训练时的 classes.txt 顺序一致）
# 示例：如果你的数据集只有 "car", "person"
class_names = ["car", "person"]  # ← 请替换成你自己的类别！

# 可选：从 classes.txt 文件自动读取
# classes_file = r"E:\tcy_data\datasets2\classes.txt"
# with open(classes_file, 'r', encoding='utf-8') as f:
#     class_names = [line.strip() for line in f.readlines()]

# ----------------------------
# 读取图像
# ----------------------------
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"无法读取图像: {image_path}")

h, w = img.shape[:2]

# ----------------------------
# 读取YOLO格式的标注文件
# 每行格式: class_id center_x center_y width height （均为归一化值）
# ----------------------------
if not os.path.exists(label_path):
    print(f"⚠️ 标注文件不存在: {label_path}")
    boxes = []
else:
    with open(label_path, 'r') as f:
        lines = f.readlines()

    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])

        # 转换为像素坐标（左上角 x1,y1 和右下角 x2,y2）
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        boxes.append((cls_id, x1, y1, x2, y2))

# ----------------------------
# 绘制边界框和标签
# ----------------------------
colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # 多类别颜色循环

for cls_id, x1, y1, x2, y2 in boxes:
    # 获取类别名
    label = class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}"

    # 选择颜色
    color = colors[cls_id % len(colors)]

    # 画矩形框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # 画标签背景
    cv2.rectangle(img, (x1, y1 - 20), (x1 + len(label) * 12, y1), color, -1)

    # 写标签文字
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# ----------------------------
# 显示结果
# ----------------------------
cv2.namedWindow("YOLO Labels Visualization", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Labels Visualization", min(w, 1200), min(h, 800))
cv2.imshow("YOLO Labels Visualization", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ----------------------------
# 可选：保存带框图像
# ----------------------------
output_path = os.path.splitext(image_path)[0] + "_with_boxes.jpg"
cv2.imwrite(output_path, img)
print(f"✅ 带标注框的图像已保存至: {output_path}")