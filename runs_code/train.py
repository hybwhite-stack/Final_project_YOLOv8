#运行前把data中的cache文件都删除

from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # 1. 确认一下显卡是否真的被识别
    print("显卡是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前使用的显卡:", torch.cuda.get_device_name(0))

    # 2. 强行只读取我们改好（带CBAM）的 yaml 文件！
    # 关键点：不要再写 yolov8n.pt，否则它会无视你的修改！
    model = YOLO('yolov8.yaml')

    # 3. 开始训练
    # 你的分辨率之前是 640，保持这个大小能让显卡跑得更稳
    results = model.train(
        data=r'D:\pythonProject\graduationProject\data.yaml',
        epochs=40,
        imgsz=640,
        batch=8,  # 你的 3060 有 6G 显存，16 没问题。如果报错 OOM (Out Of Memory)，改成 8
        device=0,  # 强行指定使用第 0 张显卡 (你的 RTX 3060)
        workers=4,

        # # 🔍 改进尝试：调节数据增强
        # mosaic=0.8,  # 将Mosaic马赛克增强的概率从1.0稍微降低，更稳健一些
        # mixup=0.1,  # 开启一点 Mixup 混合增强，增加样本多样性
        # # 🔍 改进尝试：加强正则化防过拟合
        # weight_decay=0.001,  # 将默认的 L2 正则化系数（通常是0.0005）增加一倍，惩罚过大的参数值，从而防过拟合

        project=r'D:\pythonProject\graduationProject\runs\detect',

        name='40'  # 这次训练结果保存的文件夹名字

    )
    print("恭喜！训练成功完成！")