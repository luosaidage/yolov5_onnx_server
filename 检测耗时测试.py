import time
t1 = time.perf_counter()
from PIL import Image,ImageDraw
from utils.operation import YOLO


t2 = time.perf_counter()
res = YOLO(onnx_path=r'ReqFile/leaf_det.onnx')
det = res.decect(r'ReqFile/sample_pic.JPG')

# 结果
print (det)


# 画框框
img = Image.open(r'ReqFile/sample_pic.JPG')
draw = ImageDraw.Draw(img)

for i in range(len(det)):
    draw.rectangle(det[i]['crop'],width=3)
img.show() # 展示

t3 = time.perf_counter()
print (f'从头加载耗时：{t3-t1}秒')
print (f'识别过程耗时：{t3-t2}秒')