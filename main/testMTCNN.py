import mtcnn
from PIL import Image
import matplotlib.pyplot as plt
from numpy import asarray 

print(mtcnn.__version__)

path = "../Data/test/son1.jpg"
img = Image.open(path)
# plt.imshow(img)
# plt.show()
img_array = asarray(img)
detector = mtcnn.MTCNN()
face = detector.detect_faces(img_array)
x,y,w,h = face[0]['box']
x1 = x - w*0.1 if x - w*0.1 > 0 else 0
y1 = y - h*0.1 if y - h*0.1 > 0 else 0
x2 = x1 + w*1.2 if x1 + w*1.2 < img_array.shape[1] else img_array.shape[1]
y2 = y1 + h*1.2 if y1 + w*1.2 < img_array.shape[0] else img_array.shape[0]
img_new = img_array[int(y1):int(y2), int(x1):int(x2)]
img_new = Image.fromarray(img_new)
# plt.imshow(face)
# plt.show()
# print(face)
plt.imshow(img_new)

plt.show()
