import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import dataset
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.transforms.transforms import RandomAffine
from tqdm import tqdm

input_path = "../dataset/"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])

data_transforms = {
    'train':
    transforms.Compose([

        transforms.Resize((224,224)),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomAffine(0,shear=5 , scale=(0.8,1.2)),
        transforms.RandomRotation(5),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.ColorJitter(brightness=(0.8, 1.5), contrast=(0.8,1.5), saturation=0, hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'val':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])
}

image_datasets = {
    'train':
    datasets.ImageFolder(input_path+"train/", data_transforms['train']),
    'val':
    datasets.ImageFolder(input_path+"val/", data_transforms['val'])
}

dataloaders = {
    'train':
    torch.utils.data.DataLoader(image_datasets['train'],
                                batch_size = 16,
                                shuffle = True,
                                num_workers = 4),
    'val':
    torch.utils.data.DataLoader(image_datasets['val'],
                                batch_size = 16,
                                shuffle =False,
                                num_workers=4)
}

class_names = image_datasets['train'].classes
dataset_sizes = {x:len(image_datasets[x]) for x in ['train','val']}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_resnet18 = models.resnet18(pretrained=True)
in_features = model_resnet18.fc.in_features
model_resnet18.fc = nn.Linear(in_features, len(class_names))
model_resnet18 = model_resnet18.to(device)
loss = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_resnet18.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,step_size=7, gamma = 0.1) # sau 7 epoch lr giam gamma=0.1

def train_model(model, lss, optimizer, num_epoch = 1):
    for epoch in range(num_epoch):
        print("Epoch {}/{}".format(epoch+1, num_epoch))
        print('-'*15*5)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else : 
                model.eval()
            
            running_loss = 0.0
            running_corrects =0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss =lss(outputs , labels) 

                if phase =='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs,1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
        torch.save(model.state_dict(), 'gender.pth')
    return model

if __name__ == '__main__':
    model_trained = train_model(model_resnet18, loss, optimizer_ft,num_epoch=15)