from ultralytics.utils.callbacks.base import optimizer_step

from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld

if __name__ == '__main__':
    model = (YOLO(r"D:\tcy_works\YOLOv11\ultralytics-main\ultralytics\cfg\models\v8\yolov8-CSPC.yaml")
             .load(r'D:\tcy_works\YOLOv11\ultralytics-main\load_models\yolov8n.pt'))

    results = model.train(
        data=r"D:\tcy_works\Full_stack\Django\car_van\datasets2\data.yaml",
        epochs=150,
        imgsz=640,
        batch=8,
        optimizer='Lion',
    )