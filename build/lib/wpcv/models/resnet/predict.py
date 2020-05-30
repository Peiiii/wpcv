import torch
from torchvision import transforms
import os,shutil,glob
import cv2
from PIL import Image

class Predictor:
    def __init__(self, model_path='model.pkl', input_size=(224,224),classes=None,classes_path=None,transform=None):
        self.model_path = model_path
        self.model = torch.load(model_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        transform = transform or  transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform=transform
        if classes_path:
            classes=open(classes_path).read().strip().split('\n')
        self.classes=classes
    def predict(self,img):
        im=img
        im = self.transform(im).float()
        im = torch.tensor(im, requires_grad=False)
        im = im.unsqueeze(0)
        im = im.to(self.device)
        y = self.model(im)
        y = torch.argmax(y).cpu().int()
        y = int(y)
        return self.classes[y]
    def predict_from_file(self,fp):
        img=Image.open(fp)
        return self.predict(img)
    def test_dir(self,dir,out_dir,verbose=True,font_path=None,font_size=64):
        from wpkit.cv.utils import mark_img,set_font_path
        if not font_path:
            if os.path.exists('msyh.ttf'):
                font_path='msyh.ttf'
        if font_path:
            set_font_path(font_path)
        fs = glob.glob(dir+'/*.jpg')
        res = {}
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        for i, f in enumerate(fs):
            name = os.path.basename(f)
            y = self.predict_from_file(f)
            img=Image.open(f)
            img=mark_img(img,y,font_size=font_size)
            f2=out_dir+'/'+os.path.basename(f)
            img.save(f2)
            if verbose:
                print(i, f, y)
        return res
    def predict_dir(self,dir,verbose=True):
        fs=glob.glob(dir+'/*.jpg')
        res={}
        for i,f in enumerate(fs):
            name=os.path.basename(f)
            y=self.predict_from_file(f)
            res[name]=y
            if verbose:
                print(i,f,y)
        return res



if __name__ == '__main__':
    predictor=Predictor(
        model_path='model.pkl',
        classes_path='classes.txt'
    )
    predictor.test_dir(
        '/home/ars/disk/chaoyuan/dataset/汽车分类/raw',
        out_dir="/home/ars/disk/chaoyuan/dataset/汽车分类/raw_test_results",
        font_path='arsocr/utils/msyh.ttf'
    )

