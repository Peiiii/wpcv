
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
print("PyTorch Version: ",torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)
from PIL import Image


def main():

    test(
        data_dir="/home/ars/sda5/data/chaoyuan/datasets/classify_datasets/申请表字段有无签名/val/无",
        out_dir="/home/ars/sda5/data/chaoyuan/datasets/classify_datasets/申请表字段有无签名/val_test_results",
        model_path='model.pkl',
        input_size=224,
        classes_path='trained_models/handwriting/classes.txt'
    )

def test(
    data_dir,
    out_dir=None,
    model_path = 'model.pkl',
    input_size = 224,
    classes = None,
    classes_path=None
):
    if classes_path:
        classes=open(classes_path).read().strip().split('\n')
    model, device, classes, transform=init(model_path,classes,input_size)
    test_dir(data_dir,model,out_dir,device,classes,transform)

def init(model_path,classes,input_size):


    # imgs = [data_dir + '/' + i for i in os.listdir(data_dir)]
    ###########################
    model_name = 'resnet'
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    transform = data_transforms['val']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ###########################
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    return model,device,classes,transform
###########################
os.system('killall display')
def mark_img(img,text):
    from PIL import Image,ImageDraw,ImageFont
    draw=ImageDraw.ImageDraw(img)
    font=ImageFont.truetype(font='arsocr/utils/msyh.ttf',size=16)
    draw.text((0,0),text=text,fill='red',font=font)
    return img


def test_dir(dir,model,out_dir,device,classes,transform):
    if not out_dir:
        out_dir=os.path.dirname(dir)+'/'+os.path.basename(dir)+'_test_results'
    if os.path.exists(out_dir):shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    for i,f in enumerate(glob.glob(dir+'/*.jpg')):
        f2=out_dir+'/'+os.path.basename(f)
        img=load_img(f,transform)
        img=img.to(device)
        y=model(img)
        y = torch.argmax(y).cpu().int()
        y = int(y)
        cls=classes[y]
        img=Image.open(f)
        img=mark_img(img,cls)
        img.save(f2)
        print(i,f)


def load_img(f,transform):
    im = Image.open(f)
    im = transform(im).float()
    im = torch.tensor(im, requires_grad=False)
    im = im.unsqueeze(0)
    return im

def val_dir(data_dir,model,device,classes,transform):
    def iter_data():
        for cls in os.listdir(data_dir):
            cls_dir = data_dir + '/' + cls
            for f in glob.glob(cls_dir + '/*.jpg'):
                yield f, cls

    correct = 0
    total = 0
    for i, (f, cls) in enumerate(iter_data()):
        im=load_img(f,transform)
        im = im.to(device)
        y = model(im)
        y = torch.argmax(y).cpu().int()
        y = int(y)
        # print(y,len(classes))
        pred = classes[y]
        if pred == cls:
            correct += 1
        total += 1
        # print(cls==pred)
        print(i, f, y, pred, cls)

    accuracy = correct / total
    print('total: %s , accuracy : %s' % (total, accuracy))

if __name__ == '__main__':
    # test_dir(data_dir,model)

    main()