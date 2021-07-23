import numpy as np
import torch
from torchvision import models, datasets,transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt

normalize= transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
preprocess = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize])
    
img_dataset = datasets.ImageFolder("../dataset/val/", preprocess)
class_names = img_dataset.classes
print(class_names)
model_resnet18 = models.resnet18(pretrained=True)
in_features = model_resnet18.fc.in_features
model_resnet18.fc = nn.Linear(in_features, 6)
model_resnet18.load_state_dict(torch.load('./gender.pth'))
model_resnet18.eval()

with torch.no_grad():
    path = '../dataset/test/obama.jpg'
    img_test  = Image.open(path)
    img_transform = preprocess(img_test)

    prediction = model_resnet18(img_transform.unsqueeze(0))
    predict = prediction.detach().numpy()
    print(predict)
    # Predicted class value using argmax
    predicted_class = np.argmax(predict)
    
# Show result
plt.imshow(img_test, cmap='gray')
plt.title(f'Prediction: {class_names[predicted_class]}')
plt.show()