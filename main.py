from fastapi import FastAPI, UploadFile, File

from io import BytesIO
from PIL import Image,ImageDraw
from utils.operation import YOLO


def detect(onnx_path='ReqFile/yolov5n-7-k5.onnx',img=r'ReqFile/bus.jpg',show=True):
    '''
    检测目标，返回目标所在坐标如：
    {'crop': [57, 390, 207, 882], 'classes': 'person'},...]
    :param onnx_path:onnx模型路径
    :param img:检测用的图片
    :param show:是否展示
    :return:
    '''
    yolo = YOLO(onnx_path=onnx_path)  # 加载yolo类
    det_obj = yolo.decect(img)  # 检测

    # 打印检测结果
    print (det_obj)

    # 画框框
    if show:
        img = Image.open(img)
        draw = ImageDraw.Draw(img)

        for i in range(len(det_obj)):
            draw.rectangle(det_obj[i]['crop'],width=3)
        img.show()  # 展示
    return det_obj

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/detect/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()  # 接收浏览器上传的图片
    im1 = BytesIO(contents)  # 将数据流转换成二进制文件存在内存中

    # 返回结果
    return detect(onnx_path='ReqFile/yolov5n-7-k5.onnx', img=im1, show=False)
