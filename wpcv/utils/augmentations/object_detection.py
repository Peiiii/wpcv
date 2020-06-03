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


class BboxToPoints(object):
    def __call__(self, img, bbox):
        points = np.array(bbox).reshape((-1, 2))
        return img, points


class PointsToBbox(object):
    def __call__(self, img, points):
        bbox = np.array(points).reshape((-1))
        return img, bbox


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
            points = PT.hflip(points, imw)
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
            points = PT.vflip(points, imh)
        return img, points

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
class RandomAffine(object):
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        BT.RandomAffine
class Affine(object):
    def affine(img, angle, translate, scale, shear, resample=0, fillcolor=None):
        """Apply affine transformation on the image keeping image center invariant

        Args:
            img (PIL Image): PIL Image to be rotated.
            angle (float or int): rotation angle in degrees between -180 and 180, clockwise direction.
            translate (list or tuple of integers): horizontal and vertical translations (post-rotation translation)
            scale (float): overall scale
            shear (float or tuple or list): shear angle value in degrees between -180 to 180, clockwise direction.
            If a tuple of list is specified, the first value corresponds to a shear parallel to the x axis, while
            the second value corresponds to a shear parallel to the y axis.
            resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
                An optional resampling filter.
                See `filters`_ for more information.
                If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
            fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
        """

        assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
            "Argument translate should be a list or tuple of length 2"

        assert scale > 0.0, "Argument scale should be positive"

        output_size = img.size
        center = (img.size[0] * 0.5 + 0.5, img.size[1] * 0.5 + 0.5)
        matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
        kwargs = {"fillcolor": fillcolor} if PILLOW_VERSION[0] >= '5' else {}
        print(matrix)
        return img.transform(output_size, Image.AFFINE, matrix, resample, **kwargs)


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
        RandomAffine(30)
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
    img, points = transform(img, points)
    img=wpcv.draw_polygon(img,points)
    img.show()
    # print(img.size, box)


if __name__ == '__main__':
    demo()
