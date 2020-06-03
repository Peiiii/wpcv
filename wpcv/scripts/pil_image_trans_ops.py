
import os, sys, shutil, math, random, json, multiprocessing, threading
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import abc

def pilimg(img):
    if isinstance(img,Image.Image):return img
    if isinstance(img,np.ndarray):
        if len(img.shape)==3:img=img[:,:,::-1]
    return Image.fromarray(np.array(img).astype(np.uint8))
def cv2img(img):
    if isinstance(img,Image.Image):
        img=np.array(img)
        if len(img.shape)==3:img=img[:,:,::-1]
        return img
    return img

class TransBase(object):
    def __init__(self, probability = 1.):
        super(TransBase, self).__init__()
        self.probability = probability
    @abc.abstractmethod
    def tranfun(self, inputimage):
        pass
    def process(self,inputimage):
        if np.random.random() < self.probability:
            return self.tranfun(inputimage)
        else:
            return inputimage

class RandomContrast(TransBase):
    def setparam(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."
    def tranfun(self, image):
        image = pilimg(image)
        enh_con = ImageEnhance.Brightness(image)
        return enh_con.enhance(random.uniform(self.lower, self.upper))

class RandomBrightness(TransBase):
    def setparam(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."
    def tranfun(self, image):
        image = pilimg(image)
        bri = ImageEnhance.Brightness(image)
        return bri.enhance(random.uniform(self.lower, self.upper))

class RandomColor(TransBase):
    def setparam(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."
    def tranfun(self, image):
        image = pilimg(image)
        col = ImageEnhance.Color(image)
        return col.enhance(random.uniform(self.lower, self.upper))

class RandomSharpness(TransBase):
    def setparam(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."
    def tranfun(self, image):
        image = pilimg(image)
        sha = ImageEnhance.Sharpness(image)
        return sha.enhance(random.uniform(self.lower, self.upper))

class Compress(TransBase):
    def setparam(self, lower=5, upper=85):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."
    def tranfun(self, image):
        img = cv2img(image)
        param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(self.lower, self.upper)]
        img_encode = cv2.imencode('.jpeg', img, param)
        img_decode = cv2.imdecode(img_encode[1], cv2.IMREAD_COLOR)
        pil_img = pilimg(img_decode)
        if len(image.split())==1:
            pil_img = pil_img.convert('L')
        return pil_img

class Exposure(TransBase):
    def setparam(self, lower=5, upper=10):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."
    def tranfun(self, image):
        image = cv2img(image)
        h,w = image.shape[:2]
        x0 = random.randint(0, w)
        y0 = random.randint(0, h)
        x1 = random.randint(x0, w)
        y1 = random.randint(y0, h)
        transparent_area = (x0, y0, x1, y1)
        mask=Image.new('L', (w, h), color=255)
        draw=ImageDraw.Draw(mask)
        mask = np.array(mask)
        if len(image.shape)==3:
            mask = mask[:,:,np.newaxis]
            mask = np.concatenate([mask,mask,mask],axis=2)
        draw.rectangle(transparent_area, fill=random.randint(150,255))
        reflection_result = image + (255 - mask)
        reflection_result = np.clip(reflection_result, 0, 255)
        return pilimg(reflection_result)

class Rotate(TransBase):
    def setparam(self, lower=-5, upper=5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        # assert self.lower >= 0, "lower must be non-negative."
    def tranfun(self, image):
        image = pilimg(image)
        rot = random.uniform(self.lower, self.upper)
        trans_img = image.rotate(rot, expand=True)
        return trans_img

class Blur(TransBase):
    def setparam(self, lower=0, upper=1):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "upper must be >= lower."
        assert self.lower >= 0, "lower must be non-negative."
    def tranfun(self, image):
        image = pilimg(image)
        r=random.random()*(self.upper-self.lower)+self.lower
        image = image.filter(ImageFilter.GaussianBlur(radius=r))
        return image


class Salt(TransBase):
    def setparam(self, rate=0.02):
        self.rate = rate
    def tranfun(self, image):
        image = pilimg(image)
        num_noise = int(image.size[1] * image.size[0] * self.rate)
        # assert len(image.split()) == 1
        for k in range(num_noise):
            i = int(np.random.random() * image.size[1])
            j = int(np.random.random() * image.size[0])
            image.putpixel((j, i), int(np.random.random() * 255))
        return image


class AdjustResolution(TransBase):
    def setparam(self, max_rate=0.95,min_rate = 0.5):
        self.max_rate = max_rate
        self.min_rate = min_rate

    def tranfun(self, image):
        image = pilimg(image)
        w, h = image.size
        rate = np.random.random()*(self.max_rate-self.min_rate)+self.min_rate
        w2 = int(w*rate)
        h2 = int(h*rate)
        image = image.resize((w2, h2))
        image = image.resize((w, h))
        return image


class Crop(TransBase):
    def setparam(self, maxv=2):
        self.maxv = maxv
    def tranfun(self, image):
        img = (image)
        h,w = img.shape[:2]
        org = np.array([[0,np.random.randint(0,self.maxv)],
                        [w,np.random.randint(0,self.maxv)],
                        [0,h-np.random.randint(0,self.maxv)],
                        [w,h-np.random.randint(0,self.maxv)]],np.float32)
        dst = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
        M = cv2.getPerspectiveTransform(org,dst)
        res = cv2.warpPerspective(img,M,(w,h))
        return pilimg(res)

class Crop2(TransBase):
    def setparam(self, maxv_h=4, maxv_w=4):
        self.maxv_h = maxv_h
        self.maxv_w = maxv_w
    def tranfun(self, image_and_loc):
        image, left, top, right, bottom = image_and_loc
        w, h = image.size
        left = np.clip(left,0,w-1)
        right = np.clip(right,0,w-1)
        top = np.clip(top, 0, h-1)
        bottom = np.clip(bottom, 0, h-1)
        img = cv2img(image)
        try:
            res = pilimg(img[top:bottom,left:right])
            return res
        except AttributeError as e:
            print( left, top, right, bottom)

        h = bottom - top
        w = right - left
        org = np.array([[left - np.random.randint(0, self.maxv_w), top + np.random.randint(-self.maxv_h, self.maxv_h//2)],
                        [right + np.random.randint(0, self.maxv_w), top + np.random.randint(-self.maxv_h, self.maxv_h//2)],
                        [left - np.random.randint(0, self.maxv_w), bottom - np.random.randint(-self.maxv_h, self.maxv_h//2)],
                        [right + np.random.randint(0, self.maxv_w), bottom - np.random.randint(-self.maxv_h, self.maxv_h//2)]], np.float32)
        dst = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
        M = cv2.getPerspectiveTransform(org,dst)
        res = cv2.warpPerspective(img,M,(w,h))
        return pilimg(res)

class Stretch(TransBase):
    def setparam(self, max_rate = 1.2,min_rate = 0.8):
        self.max_rate = max_rate
        self.min_rate = min_rate
    def tranfun(self, image):
        image = pilimg(image)
        w, h = image.size
        rate = np.random.random()*(self.max_rate-self.min_rate)+self.min_rate
        w2 = int(w*rate)
        image = image.resize((w2, h))
        return image
