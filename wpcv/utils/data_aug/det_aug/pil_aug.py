import numpy as np
import random
import numbers
import cv2
from PIL import Image
import wpcv
from wpcv.utils.ops import pil_ops, polygon_ops
from wpcv.utils.data_aug.base import Compose, Zip
from wpcv.utils.data_aug import img_aug


class ToPILImage(object):
    def __init__(self):
        self.to = img_aug.ToPILImage()

    def __call__(self, img, *args):
        if len(args):
            return (self.to(img), *args)
        else:
            return self.to(img)


class BboxesToPoints(object):
    def __call__(self, img, bboxes):
        points = np.array(bboxes).reshape((-1, 2, 2))
        return img, points


class PointsToBboxes(object):
    def __call__(self, img, points):
        bboxes = np.array(points).reshape((-1, 4))
        return img, bboxes


class Reshape(object):
    def __init__(self, shape):
        self.target_shape = shape

    def __call__(self, x):
        return np.array(x).reshape(self.target_shape)


class Limitsize(object):
    def __init__(self, maxsize):
        limit = maxsize
        if isinstance(limit, (tuple, list, set,)):
            mw, mh = limit
        else:
            mw = mh = limit
        self.size = (mw, mh)

    def __call__(self, img, points):
        mw, mh = self.size
        w, h = img.size
        rw = w / mw
        rh = h / mh
        r = max(rw, rh)
        if r > 1:
            nw, nh = int(w / r), int(h / r)
            img = pil_ops.resize(img, (nw, nh))
            points = polygon_ops.scale(points, 1 / r)
        return img, points


class Scale(object):
    def __init__(self, scales):
        if isinstance(scales, (tuple, list)):
            scaleX, scaleY = scales
        else:
            scaleX = scaleY = scales
        self.scaleX, self.scaleY = scaleX, scaleY

    def __call__(self, img, points):
        scaleX, scaleY = self.scaleX, self.scaleY
        img = pil_ops.scale(img, (scaleX, scaleY))
        points = polygon_ops.scale(points, (scaleX, scaleY))
        return img, points


class Resize(object):
    def __init__(self, size, keep_ratio=False, fillcolor='black'):
        self.size = size
        self.keep_ratio = keep_ratio
        self.fillcolor = fillcolor

    def __call__(self, img, points):
        w, h = img.size
        tw, th = self.size
        if not self.keep_ratio:
            scaleX, scaleY = tw / w, th / h
            img = pil_ops.resize(img, self.size)
            points = polygon_ops.scale(points, (scaleX, scaleY))
        else:
            if self.fillcolor is 'random':
                fillcolor = tuple(np.random.choice(range(256), size=3))
            else:
                fillcolor = self.fillcolor
            img = pil_ops.resize_keep_ratio(img, self.size, fillcolor=fillcolor)
            rx = w / tw
            ry = h / th
            r = max(rx, ry)
            nw = w / r
            nh = h / r
            dw = (tw - nw) // 2
            dh = (th - nh) // 2
            points = polygon_ops.scale(points, 1 / r)
            points = polygon_ops.translate(points, (dw, dh))
        return img, points


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, points):
        imw, imh = img.size
        if random.random() < self.p:
            img = pil_ops.hflip(img)
            points = [polygon_ops.hflip(pnts, imw) for pnts in points]
        return img, points

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, points):
        imw, imh = img.size
        if random.random() < self.p:
            img = pil_ops.vflip(img)
            points = [polygon_ops.vflip(pnts, imh) for pnts in points]
        return img, points

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomTranslate(object):
    def __init__(self, max_offset=None, fillcolor='black'):
        if max_offset is not None and len(max_offset) == 2:
            mx, my = max_offset
            max_offset = [-mx, -my, mx, my]
        self.max_offset = max_offset
        self.fillcolor = fillcolor

    def __call__(self, img, points):
        if self.fillcolor is 'random':
            fillcolor = tuple(np.random.choice(range(256), size=3))
        else:
            fillcolor = self.fillcolor
        rang = polygon_ops.get_translate_range(points, img.size)
        if self.max_offset:
            def limit_box(box, limits=None):
                if limits is None: return box
                if len(limits) == 2:
                    ml, mt = 0, 0
                    mr, mb = limits
                else:
                    assert len(limits) == 4
                    ml, mt, mr, mb = limits
                l, t, r, b = box
                l = max(ml, l)
                t = max(mt, t)
                r = min(mr, r)
                b = min(mb, b)
                if l > r:
                    return None
                if t > b: return None
                return [l, t, r, b]

            rang = limit_box(rang, self.max_offset)
            if rang is None:
                return img, points
        ofx = random.randint(rang[0], rang[2])
        ofy = random.randint(rang[1], rang[3])
        img = pil_ops.translate(img, offset=(ofx, ofy), fillcolor=fillcolor)
        points = [polygon_ops.translate(pnts, (ofx, ofy)) for pnts in points]
        return img, points


class RandomRotate(object):
    def __init__(self, degree, expand=True, fillcolor='black'):
        self.degree = degree if not isinstance(degree, numbers.Number) else [-degree, degree]
        self.expand = expand
        self.fillcolor = fillcolor

    def __call__(self, img, points):
        if self.fillcolor is 'random':
            fillcolor = tuple(np.random.choice(range(256), size=3))
        else:
            fillcolor = self.fillcolor
        degree = random.random() * (self.degree[1] - self.degree[0]) + self.degree[0]
        w, h = img.size
        img = pil_ops.rotate(img, degree, expand=self.expand, fillcolor=fillcolor)
        points = [polygon_ops.rotate(pnts, degree, (w // 2, h // 2), img_size=(w, h), expand=self.expand) for pnts in
                  points]
        return img, points


class RandomShearX(object):
    def __init__(self, degree):
        self.degree = degree if not isinstance(degree, numbers.Number) else [-degree, degree]

    def __call__(self, img, points):
        degree = random.random() * (self.degree[1] - self.degree[0]) + self.degree[0]
        w, h = img.size
        img = pil_ops.shear_x(img, degree)
        points = [polygon_ops.shear_x(pnts, degree, img_size=(w, h), expand=True) for pnts in points]
        return img, points


class RandomShearY(object):
    def __init__(self, degree):
        self.degree = degree if not isinstance(degree, numbers.Number) else [-degree, degree]

    def __call__(self, img, points):
        degree = random.random() * (self.degree[1] - self.degree[0]) + self.degree[0]
        w, h = img.size
        img = pil_ops.shear_y(img, degree)
        points = [polygon_ops.shear_y(pnts, degree, img_size=(w, h), expand=True) for pnts in points]
        return img, points


class RandomShear(object):
    def __init__(self, xdegree, ydegree=None, fillcolor='balck'):
        def get_param(param, defualt=None):
            if param is None: return defualt
            return param if not isinstance(param, numbers.Number) else [-param, param]

        self.xdegree = get_param(xdegree)
        self.ydegree = get_param(ydegree)
        self.fillcolor = fillcolor

    def __call__(self, img, points):

        if self.xdegree:
            if self.fillcolor is 'random':
                fillcolor = tuple(np.random.choice(range(256), size=3))
            else:
                fillcolor = self.fillcolor
            degree = random.random() * (self.xdegree[1] - self.xdegree[0]) + self.xdegree[0]
            w, h = img.size
            img = pil_ops.shear_x(img, degree, fillcolor=fillcolor)
            points = [polygon_ops.shear_x(pnts, degree, img_size=(w, h), expand=True) for pnts in points]
        if self.ydegree:
            if self.fillcolor is 'random':
                fillcolor = tuple(np.random.choice(range(256), size=3))
            else:
                fillcolor = self.fillcolor
            degree = random.random() * (self.ydegree[1] - self.ydegree[0]) + self.ydegree[0]
            w, h = img.size
            img = pil_ops.shear_y(img, degree, fillcolor=fillcolor)
            points = [polygon_ops.shear_y(pnts, degree, img_size=(w, h), expand=True) for pnts in points]
        return img, points


# class RandomPerspective:

