# FACE_RECOGNITION

## INSTALL

```
pip install -U Cython cmake numpy
pip install -U insightface
pip install onnxruntime onnxruntime-gpu 
```

## FaceRecognition：

|接口名称|参数|返回值类型|备注|
|----|----|----|----|
|__init__|face_db，root||分别为人脸数据库存储路径 (默认 ./face_db ) 以及模型加载路径 (默认 ./ ，会在该文件夹下的models文件夹内获取模型信息)|
|recognition|image|list|单个人脸信息为一个tuple，其中包括了 ( face_id , face_save_path )，其中 face_id为其在人脸库中的 id，face_save_path为其在人脸库中人脸图片的存储路径，按图片中人脸**从左到右**的顺序获取信息，如果图片中包含未注册的人脸，该人脸项为 None，如果图片中无人脸返回空列表|
|register|image|list|对图片中未注册在人脸库中的人脸进行注册，并按图片中人脸**从左到右**的顺序获取信息，对于图片中已注册过的人脸，会直接获取其在人脸库中的已有信息（tuple），如果图片中无人脸返回空列表|
|delete|image|list|对图片中已注册在人脸库中的人脸进行匹配并删除，并按图片中人脸**从左到右**的顺序获取信息，未注册的人脸项为 None，如果图片中无人脸返回空列表|
|clear_facedb|||直接清空人脸库中的所有信息（包括已存储的人脸图片）|

## Quick Example

```
import os
import time
import cv2
import insightface
import numpy as np
from facetools.FaceRecognition import FaceRecognition


zjl1_pic = cv2.imread('./pictures_face/zjl1.jpeg')
zjl2_pic = cv2.imread('./pictures_face/zjl2.jpeg')
chenhe1_pic = cv2.imread('./pictures_face/chenhe1.jpeg')
chenhe2_pic = cv2.imread('./pictures_face/chenhe2.jpeg')
people3_pic = cv2.imread('./pictures_face/c_z_e_3.png')
people4_pic = cv2.imread('./pictures_face/j_c_z_e_4.png')

print('---------------------------------------------------------------')
app = FaceRecognition()
print('---------------------------------------------------------------')

results = app.recognition(zjl1_pic)
print(results) 
# [None]

results = app.register(zjl1_pic)
print(results) 
# [(0, 'face_db/facepics/0.png')]

results = app.recognition(zjl2_pic)
print(results) 
# [(0, 'face_db/facepics/0.png')]

results = app.register(chenhe1_pic)
print(results) 
# [(1, 'face_db/facepics/1.png')]

results = app.recognition(people4_pic)
print(results) 
# [None, (1, 'face_db/facepics/1.png'), (0, 'face_db/facepics/0.png'), None]

results = app.register(people3_pic)
print(results) 
# [(1, 'face_db/facepics/1.png'), (0, 'face_db/facepics/0.png'), (2, 'face_db/facepics/2.png')]

results = app.delete(people4_pic)
print(results) 
# [None, (1, 'face_db/facepics/1.png'), (0, 'face_db/facepics/0.png'), (2, 'face_db/facepics/2.png')]

app.clear_facedb()
# 清空人脸库

results = app.recognition(people4_pic)
print(results) 
# [None, None, None, None]
```




