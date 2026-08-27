from ultralytics import YOLO
import cv2
a1 = YOLO(r'D:\pythonProject\graduationProject\runs\detect\bolt_robust_lossfunction\weights\best.pt')#加载训练好的模型
a1(
    source=r'D:\pythonProject\graduationProject\微信图片_20260325105127_151_444.jpg',
    show=True,
    save=True,
    project=r'D:\pythonProject\graduationProject\runs\detect\results', # 设定保存的主目录
    name='20260325105127_151_444',                        # 设定你想保存的文件夹名字 (替代 predict)
    exist_ok=True                                  # 设为 True 表示覆盖旧文件，不会无限生成新文件夹
)
# 新增：让程序无限期暂停，直到你按下键盘上的任意按键，窗口才会关闭
cv2.waitKey(0)
cv2.destroyAllWindows() # 养成好习惯，结束时清理所有窗口