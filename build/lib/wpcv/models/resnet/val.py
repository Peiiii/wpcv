
import torch
print("PyTorch Version: ",torch.__version__)
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os,glob,shutil
import copy
# print("PyTorch Version: ",torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)
from PIL import Image
from wpkit.utils import  remake
# data_dir = "/home/ars/disk/chaoyuan/dataset/汽车分类/颜色/car_color_dataset/val"
data_dir = "/home/ars/sda5/data/chaoyuan/datasets/classify_datasets/申请表字段有无签名/val"
# data_dir = "/home/ars/sda5/data/chaoyuan/datasets/classify_datasets/公章/val"
err_dir=data_dir+'_errors'
remake(err_dir)
classes=os.listdir(data_dir)
classes.sort()
imgs=[data_dir+'/'+i for i in os.listdir(data_dir)]
# model_path='trained_models/车型识别/model.pkl'
model_path='model.pkl'
###########################
model_name='resnet'
if model_name=='resnet':
    # input_size=(224,224)
    input_size=(150,400)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        # transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
transform=data_transforms['val']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###########################
model=torch.load(model_path)
model.to(device)
model.eval()

###########################
os.system('killall display')


def iter_data():
    for cls in os.listdir(data_dir):
        cls_dir = data_dir + '/' + cls
        for f in glob.glob(cls_dir+'/*.jpg'):
            yield f,cls

correct=0
total=0
for i,(f,cls) in enumerate(iter_data()):
    im = Image.open(f)
    im2=im
    im = transform(im).float()
    im = torch.tensor(im, requires_grad=False)
    im = im.unsqueeze(0)
    im=im.to(device)
    y = model(im)
    y = torch.argmax(y).cpu().int()
    y=int(y)
    # print(y,len(classes))
    pred=classes[y]
    if pred==cls:
        correct+=1
    else:
        f2=err_dir+'/%s'%(pred)+os.path.basename(f)
        shutil.copy(f,f2)
    total+=1
    # print(cls==pred)
    print(i,f,y,pred,cls)

accuracy=correct/total
print('total: %s , accuracy : %s'%(total,accuracy))

