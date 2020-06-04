import numpy as np
import random
import numbers
import cv2
from PIL import Image
import wpcv
from wpcv.utils.transforms import pil as IM
from wpcv.utils.transforms import points as PT
from wpcv.utils.augmentations.base import Compose, Zip, ToPILImage
import wpcv.utils.augmentations.base as BT


class BboxesToPoints(object):
    def __call__(self, img, bboxes):
        points = np.array(bboxes).reshape((-1, 2, 2))
        return img, points


class PointsToBboxes(object):
    def __call__(self, img, points):
        bboxes = np.array(points).reshape((-1,4))
        return img, bboxes


class Reshape(object):
    def __init__(self, shape):
        self.target_shape = shape

    def __call__(self, x):
        return np.array(x).reshape(self.target_shape)


class Identical(object):
    def __call__(self, *args):
        if len(args)==1:
            return args[0]
        return args


class Scale(object):
    def __init__(self, scales):
        if isinstance(scales, (tuple, list)):
            scaleX, scaleY = scales
        else:
            scaleX = scaleY = scales
        self.scaleX, self.scaleY = scaleX, scaleY

    def __call__(self, img, points):
        scaleX, scaleY = self.scaleX, self.scaleY
        img = IM.scale(img, (scaleX, scaleY))
        points = PT.scale(points, (scaleX, scaleY))
        return img, points


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, points):
        w, h = img.size
        nw, nh = self.size
        scaleX, scaleY = nw / w, nh / h
        img = IM.scale(img, (scaleX, scaleY))
        points = PT.scale(points, (scaleX, scaleY))
        return img, points


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, points):
        imw, imh = img.size
        if random.random() < self.p:
            img = IM.hflip(img)
            points = [PT.hflip(pnts, imw) for pnts in points]
        return img, points

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, points):
        imw, imh = img.size
        if random.random() < self.p:
            img = IM.vflip(img)
            points = [PT.vflip(pnts, imh) for pnts in points]
        return img, points

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomTranslate(object):
    def __init__(self,max_offset=None,fillcolor='black'):
        if max_offset is not None and len(max_offset)==2:
            mx,my=max_offset
            max_offset=[-mx,-my,mx,my]
        self.max_offset=max_offset
        self.fillcolor=fillcolor
    def __call__(self, img,points):
        rang=PT.get_translate_range(points,img.size)
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
            rang=limit_box(rang,self.max_offset)
            if rang is None:
                return img,points
        ofx=random.randint(rang[0],rang[2])
        ofy=random.randint(rang[1],rang[3])
        img=IM.translate(img,offset=(ofx,ofy),fillcolor=self.fillcolor)
        points=[PT.translate(pnts,(ofx,ofy)) for pnts in points]
        return img,points

class RandomRotate(object):
    def __init__(self,degree,expand=True):
        self.degree=degree if not isinstance(degree,numbers.Number) else [-degree,degree]
        self.expand=expand
    def __call__(self,img,points):
        degree=random.random()*(self.degree[1]-self.degree[0])+self.degree[0]
        w, h = img.size
        img=IM.rotate(img,degree,expand=self.expand)
        points=[PT.rotate(pnts,degree,(w//2,h//2),img_size=(w,h),expand=self.expand) for pnts in points]
        return img,points

class RandomShearX(object):
    def __init__(self,degree):
        self.degree=degree if not isinstance(degree,numbers.Number) else [-degree,degree]
    def __call__(self,img,points):
        degree = random.random() * (self.degree[1] - self.degree[0]) + self.degree[0]
        w, h = img.size
        img = IM.shear_x(img, degree)
        points = [PT.shear_x(pnts, degree, img_size=(w, h), expand=True) for pnts in points]
        return img, points
class RandomShearY(object):
    def __init__(self,degree):
        self.degree=degree if not isinstance(degree,numbers.Number) else [-degree,degree]
    def __call__(self,img,points):
        degree = random.random() * (self.degree[1] - self.degree[0]) + self.degree[0]
        w, h = img.size
        img = IM.shear_y(img, degree)
        points = [PT.shear_y(pnts, degree, img_size=(w, h), expand=True) for pnts in points]
        return img, points
class RandomShear(object):
    def __init__(self,xdegree,ydegree=None):
        def get_param(param,defualt=None):
            if param is None:return defualt
            return param if not isinstance(param,numbers.Number) else [-param,param]
        self.xdegree=get_param(xdegree)
        self.ydegree=get_param(ydegree)


    def __call__(self,img,points):
        if self.xdegree:
            degree = random.random() * (self.xdegree[1] - self.xdegree[0]) + self.xdegree[0]
            w, h = img.size
            img = IM.shear_x(img, degree)
            points = [PT.shear_x(pnts, degree, img_size=(w, h), expand=True) for pnts in points]
        if self.ydegree:
            degree = random.random() * (self.ydegree[1] - self.ydegree[0]) + self.ydegree[0]
            w, h = img.size
            img = IM.shear_y(img, degree)
            points = [PT.shear_y(pnts, degree, img_size=(w, h), expand=True) for pnts in points]
        return img, points

class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image And int for grayscale) for the area
            outside the transform in the output image.(Pillow>=5.0.0)

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                    (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = [random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [random.uniform(shears[0], shears[1]),
                         random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img,points):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        img=BT.F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)
        return img,points

def demo():
    transform = Compose([
        # BboxToPoints(),
        Resize((512, 512)),
        Zip([
            Compose([
                BT.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
            ]),
            Identical()
        ]),
        # RandomRotate(30),
        # RandomShear(30,30),
        # RandomTranslate(max_offset=[100,100]),
        # RandomHorizontalFlip(),
        # RandomVerticalFlip(),
        # PointsToBbox()
    ])
    img = Image.open('/home/ars/图片/2019011816055827.jpg')

    points=[[
          255.7142857142857,
          243.1428571428571
        ],
        [
          305.7142857142857,
          243.1428571428571
        ],
        [
          310.0,
          306.4761904761905
        ],
        [
          254.28571428571428,
          305.04761904761904
        ]]
    img, points = transform(img, [points])
    # print(points)
    img=wpcv.draw_polygon(img,points[0])
    img.show()
    # print(img.size, box)


if __name__ == '__main__':
    demo()
