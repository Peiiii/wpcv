import cv2
import numpy as np
import os,shutil,glob
from wpkit.utils import remake
# from arsocr.utils.box_utils import  limit_box,pad_box
# from arsocr.utils.imtools import cv2img,pilimg
# from arsocr.utils import basic_utils as ut
from wpcv import utils as ut
from wpcv import limit_box,pad_box,cv2img
'''水平投影'''


def getHProjection(image):
    hProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0] * h
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
    # 绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y, x] = 255
    # cv2.imshow('hProjection2', hProjection)

    return h_


def getVProjection(image):
    vProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像宽度一致的数组
    w_ = [0] * w
    # 循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
    # 绘制垂直平投影图像
    for x in range(w):
        for y in range(h - w_[x], h):
            vProjection[y, x] = 255
    # cv2.imshow('vProjection',vProjection)
    return w_

def get_text_box(img,lower_binary_thresh=150,min_text_h = 10,min_thresh=10):
    boxes=[]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # retval, img = cv2.threshold(img, lower_binary_thresh, 255, cv2.THRESH_BINARY_INV)
    retval, img = cv2.threshold(img, lower_binary_thresh, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    (h, w) = img.shape
    # 水平投影
    H = getHProjection(img)
    start = 0
    thresh = min_thresh
    H_Start = []
    H_End = []
    # 根据水平投影获取垂直分割位置
    for i in range(len(H)):
        if H[i] > thresh and start == 0:
            H_Start.append(i)
            start = 1
        if H[i] <= thresh and start == 1:
            H_End.append(i)
            start = 0
    if start == 1:
        H_End.append(h)
    # 分割行，分割之后再进行列分割并保存分割位置
    # print(H_Start, H_End)

    for i in range(len(H_Start)):
        # 获取行图像
        # print(img.shape,H_Start[i],H_End[i], w)
        # if not len(H_End)>i:
        #     hend=h
        # else:
        #     hend=H_End[i]
        # cropImg = img[H_Start[i]:hend, 0:w]
        # print(H_End[i] - H_Start[i])
        if H_End[i] - H_Start[i] < min_text_h:
            continue
        boxes.append([0,H_Start[i],w,H_End[i]])
    if len(boxes):
        boxes=sorted(boxes,key=lambda box:box[3]-box[1],reverse=True)[:1]
    return boxes
def get_text_area(img,pad_ratio=0.4):
    pim=img.copy()
    img=cv2img(img)
    boxes = get_text_box(img, lower_binary_thresh=150, min_thresh=15)
    boxes = [limit_box(pad_box(box, pad_ratio=pad_ratio), img.shape[:2][::-1]) for box in boxes]
    if len(boxes):
        img=pim.crop(boxes[0])
    else:
        img=pim
    return img
def demo():
    dir="/home/ars/disk/work/超远/chaoyuan-part1-dev/data/results/bxd-tmp/2"
    fs=glob.glob(dir+'/*.jpg')
    fs.sort()
    from wpkit.cv.utils import ImageSaver
    saver=ImageSaver(save_dir='/home/ars/disk/work/超远/chaoyuan-part1-dev/data/results/tests/0',remake_dir=True)

    for i,f in enumerate(fs):
        img=cv2.imread(f)
        boxes=get_text_box(img,lower_binary_thresh=150,min_thresh=15)
        boxes=[ut.limit_box(ut.pad_box(box,pad_ratio=0.4),img.shape[:2][::-1]) for box in boxes]
        img=ut.draw_boxes(ut.pilimg(img),boxes)
        saver.save(img)
        print(i,f)





if __name__ == "__main__":

    demo()
