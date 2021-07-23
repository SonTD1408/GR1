import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("../dataset/test/taylor.jpg")
plt.imshow(img)
plt.title("Predict: TaylorSwift")
plt.show()