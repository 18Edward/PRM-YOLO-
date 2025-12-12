from ultralytics import YOLO

model = YOLO(r'E:\tcy_data\nn\YOLOv11\ultralytics-main\runs\detect\train64\weights\best.pt')
result = model.predict(
    source=r"E:\tcy_data\datasets2\images\train\pic_382.jpg",
    save=True
)

