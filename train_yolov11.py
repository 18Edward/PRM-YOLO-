from sympy.physics.units import momentum
from torch import nn
from ultralytics.utils.callbacks.base import optimizer_step

from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld

if __name__ == '__main__':
    model = (YOLO(r"ultralytics/cfg/models/11/yolo11-imp37.yaml")
             .load(r'D:\tcy_works\YOLOv11\ultralytics-main\runs\detect\train50\weights\best.pt'))
             # .load(r'D:\tcy_works\Full_stack\Django\car_van\yolo11s.pt'))
    # model.model.apply(init_weights)
    results = model.train(
        data=r'D:\tcy_works\Full_stack\Django\car_van\datasets2\data.yaml',
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