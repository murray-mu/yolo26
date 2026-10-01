from ultralytics import YOLO


def main():
    model = YOLO(model=r"runs/detect/train/weights/best.pt")  # 加载你训练好的权重文件

    # 去一张图片用来预测
    results = model(r"datasets/coco8/images/val/000000000061.jpg", show=True, save=True)
    results[0].show()  # 展示结果


if __name__ == "__main__":
    main()
