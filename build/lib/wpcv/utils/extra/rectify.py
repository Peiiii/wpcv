import cv2 as cv
import cv2
import numpy as np
import math
import os,shutil,glob
file_index=0
def save_to_dir(img,name=None,dir=None):
    global file_index
    name=name or '%s.jpg'%(file_index)
    dir = dir or '/home/ars/disk/work/超远/chaoyuan-part1-dev/data/tmp'
    name=dir+'/'+name
    cv2.imwrite(name,img)
    file_index+=1
# def line_detection(image):
#     gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     # apertureSize做Canny时梯度窗口的大小
#     edges = cv.Canny(gray, 50, 150, apertureSize=3)
#     # 返回的是r和theta
#     lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
#     for line in lines:
#         print(type(line))
#         rho, theta = line[0]
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         # 乘以1000，是根据源码乘的，通过x1、x2、y1、y2画一条直线
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * a)
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * a)
#         cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 2是所画直线长度的宽
#     cv.imshow('image_lines', image)

def show(img):
    from PIL import Image
    Image.fromarray(img).show()

def detect_angle(img):
    lines=detect_lines(img)
    if not len(lines):return None
    def angle(line):
        x1,y1,x2,y2=line
        (x1,y1),(x2,y2)=sorted([[x1,y1],[x2,y2]],key=lambda p:p[0])
        ang=math.atan2(y1-y2,x2-x1)*180/math.pi
        ang=(ang+45)%90-45
        return ang
    angles=[angle(line) for line in lines]
    # print(lines)
    # print(angles)
    angle=sum(angles)/len(angles)
    return angle
def rotate(image, angle):
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)
    # 设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    return cv2.warpAffine(image, M, (nW, nH))
def rectify(img):
    # from wpkit.cv.transform.opencv import rotate
    angle=detect_angle(img)
    # print(angle)
    if abs(angle)>1:
        img=rotate(img,-angle)
    return img
def detect_lines(image,show=False):
    h,w=image.shape[:2]
    min_length=int(min(w,h)*0.2)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 做Canny时梯度窗口的大小
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    # minLineLength是线段最小长度，maxLineGap是线段最大间隔，该间隔之内的认为是一条直线
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=min_length, maxLineGap=10)
    lines=[line[0] for line in lines]
    if not show:return lines
    for line in lines:
        # print(type(line))
        print(line)
        # （x1,y1）,(x2,y2)是线段的两点，只画出了一个个线段
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow('image_lines', image)



def demo():
    dir='/home/ars/disk/chaoyuan/数据整理/ocr测试图片/安检报告'
    dir='/home/ars/disk/chaoyuan/testdata/0429/ocr测试图片/申请表'
    dir='/home/ars/disk/chaoyuan/数据整理/ocr测试图片/在用车检验'
    fs=glob.glob(dir+'/*.jpg')
    for i,f in enumerate(fs):
        img=cv2.imread(f)
        img=rectify(img)
        save_to_dir(img,name=os.path.basename(f))

if __name__ == '__main__':
    demo()