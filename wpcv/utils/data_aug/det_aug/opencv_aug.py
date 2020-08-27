import numpy as np
import random
import numbers
import cv2
from PIL import Image
from wpcv.utils.ops import opencv_ops, polygon_ops
from wpcv.utils.data_aug import img_aug
from wpcv.utils import imutils



class ToOpencvImage(object):

    def __call__(self, img, *args):
        if len(args):
            return (imutils.cv2img(img), *args)
        else:
            return imutils.cv2img(img)


class RandomPerspective(object):
    def __init__(self, distortion_scale):
        if isinstance(distortion_scale,numbers.Number):
            distortion_scale=[0,distortion_scale]
        assert distortion_scale[0] >= 0 and distortion_scale[1] <= 1
        self.distortion_scale = distortion_scale

    def __call__(self, img, polygons):
        # img = imutils.cv2img(img)
        imh, imw = img.shape[:2]
        l, t, r, b = polygon_ops.bounding_rect(np.array(polygons).reshape((-1, 2)))
        dl = (l - 0)
        dt = (t - 0)
        dr = (imw - r)
        db = (imh - b)
        # quad = np.array([l, t, r, t, r, b, l, b]).astype(np.float)
        quad = np.array([0,0,imw,0,imw,imh,0,imh]).astype(np.float)
        distortion_scale=self.distortion_scale[0]+np.random.random(8)*(self.distortion_scale[1]-self.distortion_scale[0])
        quad += np.array([-dl, -dt, +dr, -dt, +dr, +db, -dl, +db])*distortion_scale*-1
        quad = quad.reshape((-1, 2)).astype(np.float32)
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad
        w, h = ((x1 - x0 + x2 - x3) // 2, (y3 - y0 + y2 - y1) // 2)
        w, h = int(w), int(h)
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]]).astype(np.float32)
        M = cv2.getPerspectiveTransform(quad, dst)
        img = cv2.warpPerspective(img, M, (w, h))
        tmp = []
        for polygon in polygons:
            tmp_polygon = np.zeros((len(polygon), 3))
            tmp_polygon[:, :2] = polygon
            tmp_polygon[:, 2] = 1
            tmp_polygon = tmp_polygon.dot(M.T)
            polygon = tmp_polygon[:, :2] / np.expand_dims(tmp_polygon[:, 2], -1)
            tmp.append(polygon)
        polygons = np.array(tmp)
        # img = imutils.pilimg(img)
        return img, polygons

