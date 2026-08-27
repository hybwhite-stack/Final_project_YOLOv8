from ultralytics import YOLO
import torch

if __name__ == '__main__':

    # 1. 确认一下显卡是否真的被识别
    print("显卡是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("当前使用的显卡:", torch.cuda.get_device_name(0))

    # 2. 加载我们带有 CBAM 注意力的模型
    model = YOLO('yolov8.yaml')

    # 3. 开始训练（小样本防过拟合终极版）
    results = model.train(
        data=r'D:\pythonProject\graduationProject\data.yaml',
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        workers= 4,

        # # 🔥 绝招 1：标签平滑 (有效缓解模型过度自信，防过拟合)
        # label_smoothing=0.1,
        #
        # # 🔥 绝招 2：Dropout 随机失活 (强迫网络学习更多样的特征)
        # dropout=0.1,
        #
        # # 🔥 绝招 3：余弦退火学习率 (让寻优过程更平滑，模型更稳健)
        # cos_lr=True,
        #
        # # 🎯 辅助改进：加大边界框回归的惩罚力度，让框画得更准
        # box=10.0,
        #
        # project=r'D:\pythonProject\graduationProject\runs\detect',
        name='deeeee'
    )

    print("完成")
    # Label Smoothing (Effectively mitigates model overconfidence and prevents overfitting)
    label_smoothing = 0.1

    # Dropout Random Deactivation (Forcing the Network to Learn More Diverse Features)
    dropout = 0.1,

    # Cosine Annealing Learning Rate (Makes the optimization process smoother and the model more stable) cos_lr=True,
    cos_lr = True,

    # Auxiliary improvement: Increase the penalty for bounding box regression to make the boxes drawn more accurately
    box = 10.0,