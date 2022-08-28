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