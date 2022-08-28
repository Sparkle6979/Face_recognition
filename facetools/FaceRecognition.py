from enum import EnumMeta
import os,shutil

import cv2
import insightface
import numpy as np
from sklearn import preprocessing

class FaceRecognition:
    def __init__(self, gpu_id=0, face_db='face_db', root = './', threshold=1.24, det_thresh=0.50, det_size=(640, 640)):
        """
        人脸识别工具类
        :param gpu_id: 正数为GPU的ID，负数为使用CPU
        :param face_db: 人脸库文件夹
        :param threshold: 人脸识别阈值
        :param det_thresh: 检测阈值
        :param det_size: 检测模型图片大小
        """
        self.gpu_id = gpu_id
        self.face_db = face_db
        self.threshold = threshold      # 1.24
        self.det_thresh = det_thresh    # 0.50
        self.det_size = det_size        # (640,640)
        self.root = root
        # 加载人脸识别模型，当allowed_modules=['detection', 'recognition']时，只单纯检测和识别
        self.model = insightface.app.FaceAnalysis(root=self.root,
                                                  allowed_modules=['detection','recognition'],
                                                  providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=self.gpu_id, det_thresh=self.det_thresh, det_size=self.det_size)
        # 人脸库的人脸特征
        self.info_embedding = list()
        self.now_id = 0
        # 加载人脸库中的人脸
        self.load_faces(self.face_db)

    # 加载人脸库中的人脸
    def load_faces(self, face_db_path):
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)
        facefile = os.path.join(face_db_path,'faceinfo.npy')
        if os.path.exists(facefile):
            self.info_embedding = list(np.load(facefile,allow_pickle=True))
            for embedding in self.info_embedding:
                self.now_id = max(self.now_id,embedding["id"])
            self.now_id += 1
        
    # 人脸识别
    # 检查图片下的所有人脸，如果存在返回信息，不存在返回空
    def recognition(self, image):
        faces = self.model.get(image)
        results = list()

        if len(faces) == 0: return results
        
        faces.sort(key=lambda x:x["bbox"][0])

        for face in faces:
            # 开始人脸识别
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            findface = False
            for com_face in self.info_embedding:
                r = self.feature_compare(embedding, com_face["feature"], self.threshold)
                if r:
                    user_mes = (com_face["id"] , com_face["save_path"])
                    results.append(user_mes)
                    findface = True
                    break
            if not findface:    results.append(None)
        return results

    @staticmethod
    def feature_compare(feature1, feature2, threshold):
        diff = np.subtract(feature1, feature2)
        dist = np.sum(np.square(diff), 1)
        if dist < threshold:
            return True
        else:
            return False

    def register(self, image):
        faces = self.model.get(image)
        result = list()

        if len(faces) == 0: return result
        # 判断人脸是否存在

        facepicfile = os.path.join(self.face_db,'facepics')
        os.makedirs(facepicfile,exist_ok=True)

        faces.sort(key=lambda x:x["bbox"][0])

        for face in faces:
            
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            findface = False
            for com_face in self.info_embedding:
                r = self.feature_compare(embedding, com_face["feature"], self.threshold)
                if r:
                    user_mes = (com_face["id"] , com_face["save_path"])
                    result.append(user_mes)
                    findface = True
                    break
            if findface:    continue
            # 符合注册条件保存图片，同时把特征添加到人脸特征库中
            facebbox = np.array(face.bbox).astype(np.int32).tolist()
            savepath = os.path.join(facepicfile,'%s.png'%(self.now_id))
            # 剪切图片中的单个人脸
            singleface = image[facebbox[1]:facebbox[3],facebbox[0]:facebbox[2],:]
            cv2.imencode('.png', singleface)[1].tofile(savepath)

            self.info_embedding.append({
                "id": self.now_id,
                "feature": embedding,
                "save_path": savepath
                
            })
            result.append((self.now_id,savepath))
            self.now_id += 1
            # 特征添加到人脸特征库中
        np.save(os.path.join(self.face_db,'faceinfo.npy'),self.info_embedding,allow_pickle=True)
        return result

    # 检测人脸
    def detect(self, image):
        faces = self.model.get(image)
        results = list()

        if len(faces) == 0: return results

        faces.sort(key=lambda x:x["bbox"][0])

        for face in faces:
            result = dict()
            # 获取人脸属性
            result["bbox"] = np.array(face.bbox).astype(np.int32).tolist()
            result["kps"] = np.array(face.kps).astype(np.int32).tolist()
            result["det_score"] = np.array(face.det_score).astype(np.float32).tolist()
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            result["embedding"] = embedding
            results.append(result)
        return results

    # 直接清空并删除 face_db 文件夹
    def clear_facedb(self):
        self.now_id = 0
        self.info_embedding.clear()
        if not os.path.exists(self.face_db):    return  
        shutil.rmtree(self.face_db)



    def delete(self,image):
        faces = self.model.get(image)
        results = list()
        
        if len(faces) == 0: return results

        faces.sort(key=lambda x:x["bbox"][0])

        for face in faces:
            # 开始人脸识别
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            findface = False

            for (i,com_face) in enumerate(self.info_embedding):
                r = self.feature_compare(embedding, com_face["feature"], self.threshold)
                if r:
                    user_mes = (com_face["id"] , com_face["save_path"])
                    findface = True
                    results.append(user_mes)
                    
                    self.info_embedding.pop(i)
                    picpath = com_face["save_path"]
                    if os.path.exists(picpath):
                        os.remove(picpath)
                    break
            if not findface:    results.append(None)

        np.save(os.path.join(self.face_db,'faceinfo.npy'),self.info_embedding,allow_pickle=True)
        return results