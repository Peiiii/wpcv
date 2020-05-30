import os, glob
import cv2
import numpy as np
from PIL import Image
# from arsocr.utils.extra.rectify import rectify,detect_angle,rotate
# from arsocr.utils.allutils import crop_quad,organize_points
# from wpkit.cv.ocr.utils import crop_quad,organize_points
# from wpkit.cv.table.rectify import rectify,detect_angle,rotate
# from wpkit.cv import utils as cvutils
# from wpkit.cv.utils import ImageSaver
# from wpkit.pjtools import Timer

# imsaver=ImageSaver(save_dir=r'/home/ars/disk/work/超远/chaoyuan-part1-dev/data/results/tmp',remake_dir=True)
# imsaver.deactive()
# def save_to_dir(img,name=None):
#     return imsaver.save(img,name)

from wk.cv import utils as cvutils

def get_table_contour(img,exclude_paper_edge_thresh=20):
    n=4
    # src = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    img_blur=img
    img = cv2.bitwise_not(img)

    img_inverse=img
    # save_to_dir(img)
    AdaptiveThreshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

    # mask=AdaptiveThreshold
    mask = AdaptiveThreshold
    # save_to_dir(mask)

    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = list(filter(lambda cont: cv2.contourArea(cont) > 30, contours))
    num = min(n, len(contours))
    contours = contours[:num]

    # 上面得到的contour可能是单个轮廓都是不完整的，但是叠加在一起有完整的，因此从第一次的contours  image中重新findContour
    blank=np.zeros_like(img_blur)
    img_contours=cv2.drawContours(blank,contours,-1,255,3)

    # save_to_dir(img_contours)
    contours, hierarchy = cv2.findContours(img_contours, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = list(filter(lambda cont: cv2.contourArea(cont) > 30, contours))
    num = min(n, len(contours))
    contours = contours[:num]


    ones=np.ones_like(img_blur)
    thresh=exclude_paper_edge_thresh
    # return contours[0]
    for i in range(len(contours)):
        mask=cv2.fillPoly(ones.copy(),[contours[i]],0)
        im=img_blur*mask
        mean=im.mean()
        if mean>thresh:
            return contours[i]
    if len(contours):
        return contours[0]
    return None
def get_table(img,max_approx_poly_distance=200,exclude_paper_edge_thresh=20,scale=1,src_img=None):
    # T = Timer(mute=True)
    src_img = src_img or img.copy()

    if not scale==1:
        h,w=img.shape[:2]
        h,w=int(scale*h),int(scale*w)
        img=cv2.resize(img,(w,h))

    #
    contour=get_table_contour(img,exclude_paper_edge_thresh=exclude_paper_edge_thresh)

    if contour is None:
        return None
    # im=cv2.drawContours(img.copy(),[contour],-1,[255,0,0],5)
    # save_to_dir(im)
    approx=cv2.approxPolyDP(contour,max_approx_poly_distance,True)

    # print(approx.shape)

    if len(approx)==4:
        approx=[p[0] for p in approx]
        quad=cvutils.organize_quad_points(approx)
        quad=np.array(quad)/scale
        # table=crop_quad(src_img,box)
    else:
        rect=cv2.boundingRect(approx)
        rect=np.array(rect)/scale
        # print(rect)
        x1,y1,w,h=rect
        quad=[[x1,y1],[x1+w,y1],[x1+w,y1+h],[x1,y1+h]]
    return np.array(quad).astype(np.int)
        # table=src_img[y1:y1+h,x1:x1+w]

    # save_to_dir(table)
    # return table
def cv_imread(f):
    from wpkit.cv.utils import cv2img
    return cv2img(Image.open(f))
def demo():
    # from wpkit.pjtools import run_timer,Timer
    imsaver.deactive()
    # dir = '/home/ars/disk/chaoyuan/数据整理/ocr测试图片/exp/imgs'
    # dir = r'/home/ars/disk/chaoyuan/数据整理/ocr测试图片/安检报告'
    # dir = r'/home/ars/disk/chaoyuan/数据整理/ocr测试图片/申请表'
    # dir = r'/home/ars/disk/chaoyuan/数据整理/ocr测试图片/在用车检验'
    dir = r'/home/ars/disk/chaoyuan/数据整理/ocr测试图片/保险单'
    fs = glob.glob(dir + '/*.jpg')
    # fs=['']
    T= Timer()
    for i, f in enumerate(fs):
        assert os.path.exists(dir)
        assert os.path.exists(f)
        img = cv_imread(f)
        src_img=img.copy()

        scale=0.4
        img=cvutils.transform.rescale(img,scale)
        # img = rectify(img)
        # img=rectify(img)
        angle=detect_angle(img)
        if angle and abs(angle)>1:
            img=rotate(img,-angle)
            src_img=rotate(src_img,-angle)

        quad=get_table(img)
        quad=(quad/scale).astype(np.int)
        table=crop_quad(src_img,quad)
        save_to_dir(table)
        T.step()
    print(T.mean())
if __name__ == '__main__':

    # corner()
    demo()

