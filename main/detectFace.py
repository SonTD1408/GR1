import mtcnn
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import os

def detect_face(path):
    img = Image.open(path,"r")
    if img.mode in ("RGBA", "P"): img = img.convert("RGB")
    img_array = asarray(img)
    detector = mtcnn.MTCNN()
    face = detector.detect_faces(img_array)
    if (face):
        x,y,w,h = face[0]['box']
        x1 = x - w*0.1 if x - w*0.1 > 0 else 0
        y1 = y - h*0.1 if y - h*0.1 > 0 else 0
        x2 = x1 + w*1.2 if x1 + w*1.2 < img_array.shape[1] else img_array.shape[1]
        y2 = y1 + h*1.2 if y1 + w*1.2 < img_array.shape[0] else img_array.shape[0]
        img_new = img_array[int(y1):int(y2), int(x1):int(x2)]
        img_new = Image.fromarray(img_new)
        return img_new
    else :
        return 0

def detect_all_image(path):
    train_folder = path+"train/"
    val_folder = path+"val/"

    train_img_list = []
    val_img_list = []

    list_dir_train = os.listdir(train_folder)
    list_dir_val = os.listdir(val_folder)

    for dir in list_dir_train:
        count=0
        for img_dir in os.listdir(train_folder+"/" +dir):
            img_path = train_folder+dir+"/"+img_dir
            face = detect_face(img_path)
            if (face):
                train_img_list.append(face)
                face.save("../dataAfterDetect/train/"+dir+str(count)+".jpg")
                count+=1
    for dir in list_dir_val:
        count=0
        for img_dir in os.listdir(val_folder+"/"+dir):
            img_path = val_folder+dir+"/"+img_dir
            face = detect_face(img_path)
            if(face):
                val_img_list.append(face)
                face.save("../dataAfterDetect/val/"+dir+str(count)+".jpg")
                count+=1

    return train_img_list, val_img_list

train_img_list, val_img_list = detect_all_image("../Data/")