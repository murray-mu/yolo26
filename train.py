from ultralytics import YOLO

def main():
    model = YOLO(model=r'ultralytics/cfg/models/26/yolo26n.yaml')
    model.load('yolo26n.pt') 
    # 加载预训练权重 也可以写成model.YOLO('yolo26n.pt') 如果训练自己的数据集不用加载

    # Train the model on the COCO8 dataset for 100 epochs 训练100轮
    model.train(
        data=r"datasets/coco8/coco8.yaml",  # Path to dataset configuration file 就是数据集下面的配置文件
        epochs=50,  # Number of training epochs
        imgsz=640,  # Image size for training
        batch=2,    # 批量大小
    )

    # 如果仅仅是训练 上面的已经足够了————————————————————————————————————————————————————————————————————————————————————
    # Evaluate the model's performance on the validation set    评估模型
    metrics = model.val()

    # Perform object detection on an image   去一张图片用来验证
    results = model(r"datasets/coco8/images/val/000000000036.jpg",save=True)  # Predict on an image
    results[0].show()  # Display results    展示结果
if __name__ == '__main__':
    main()