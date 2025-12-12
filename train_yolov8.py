import warnings
warnings.simplefilter("ignore", DeprecationWarning)  # 所以弃用警告均不打印
from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld

if __name__ == '__main__':
    model = (YOLO(r"E:\tcy_data\nn\YOLOv11\ultralytics-main\ultralytics\cfg\models\v8\yolov8.yaml")
             .load(r'E:\tcy_data\nn\YOLOv11\ultralytics-main\load_models\yolov8n.pt'))
             # .load(r'D:\tcy_works\Full_stack\Django\car_van\yolo11s.pt'))
    # model.model.apply(init_weights)
    results = model.train(
        data=r"E:\tcy_data\datasets2\data.yaml",
        epochs=60,
        imgsz=640,
        batch=4,
        # batch=16,

        # --- 优化器 ---
        optimizer='AdamW',
        # lr0=1e-4,
        # lrf=1e-7,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # --- 训练策略 ---
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        label_smoothing=0.1,
        close_mosaic=15,

        # --- 其他 ---
        patience=10,
        verbose=True,
        device='cuda:0',  # GPU更快

        # 降低运行显存
        mosaic = 0,
        mixup = 0,
    )