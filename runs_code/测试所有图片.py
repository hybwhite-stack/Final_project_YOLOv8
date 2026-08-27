from ultralytics import YOLO

# 1. 加载你训练好的模型
model = YOLO(r'D:\pythonProject\graduationProject\runs\detect\bolt_robust_lossfunction\weights\best.pt')

# 2. 一键批量预测
model(
    # 👇 关键修改：这里直接填入包含你所有测试图片的“文件夹路径”
    source=r'D:\pythonProject\graduationProject\图片',

    show=False,  # 批量处理时建议设为 False，让它在后台静默光速处理
    save=True,  # 设为 True，它会自动把所有画好框的图存下来

    project=r'D:\pythonProject\graduationProject\runs\detect\results',
    name='robust'
         ,  # 所有处理完的图片都会集中保存在这个文件夹里
    exist_ok=True
)

print("🎉 批量预测全部完成！快去 batch_250 文件夹里验收成果吧！")