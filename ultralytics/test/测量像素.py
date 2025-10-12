import cv2
import numpy as np
import os


import cv2
import numpy as np
import os

# 全局变量
points = []
img_copy = None

def click_event(event, x, y, flags, param):
    global points, img_copy  # ✅ 必须声明！
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"已添加点: ({x}, {y})")
        cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img_copy, f'{len(points)}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Image", img_copy)

        if len(points) == 4:
            calculate_rectangle()

def calculate_rectangle():
    global points, img_copy, img
    pts = np.array(points, dtype="float32")
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width = int(np.linalg.norm(tr - tl))
    height = int(np.linalg.norm(tr - br))

    h_img, w_img = img.shape[:2]
    width_ratio = (width / w_img) * 100
    height_ratio = (height / h_img) * 100

    print("\n=== 矩形尺寸信息 ===")
    print(f"宽度: {width} 像素 ({width_ratio:.2f}% of image width)")
    print(f"高度: {height} 像素 ({height_ratio:.2f}% of image height)")

    display_img = img_copy.copy()
    cv2.polylines(display_img, [np.int32(rect)], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.putText(display_img, f"W: {width}px ({width_ratio:.1f}%)",
                (int(tl[0]), int(tl[1] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(display_img, f"H: {height}px ({height_ratio:.1f}%)",
                (int(tl[0]), int(tl[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Image", display_img)


# 辅助函数：对四个点进行排序（左上、右上、右下、左下）
def order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]  # 左上角: x+y 最小
    br = pts[np.argmax(s)]  # 右下角: x+y 最大
    tr = pts[np.argmin(diff)]  # 右上角: x-y 最小
    bl = pts[np.argmax(diff)]  # 左下角: x-y 最大

    return np.array([tl, tr, br, bl], dtype="float32")


# 主程序
if __name__ == "__main__":
    # 读取图像（替换为你自己的图片路径）
    image_path = r"D:\tcy_works\data\DJI\yolo_data\范家新村广场分割1\pic_78.jpg"

    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 文件不存在：{image_path}")
        print(f"当前工作目录：{os.getcwd()}")
        exit()

    # 检查是否可读
    if not os.access(image_path, os.R_OK):
        print(f"❌ 文件不可读：{image_path}")
        exit()

    print(f"尝试读取的图像路径: {image_path}")
    print(f"当前工作目录: {os.getcwd()}")

    # 支持中文路径的读取方式
    try:
        with open(image_path, 'rb') as f:
            data = f.read()
        np_data = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        if img is None:
            raise Exception("解码失败")
    except Exception as e:
        print(f"❌ 无法读取或解码图像：{image_path}")
        print(f"错误信息：{e}")
        exit()

    img_copy = img.copy()
    h, w = img.shape[:2]
    print(f"✅ 图像加载成功，尺寸: {w} x {h}")

    cv2.imshow("Image", img_copy)
    cv2.setMouseCallback("Image", click_event)

    print("\n📌 请在图像上点击4个角点（建议顺序：左上 → 右上 → 右下 → 左下）")
    print("💡 提示：点击4个点后自动计算尺寸。按 'q' 键退出程序。")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()