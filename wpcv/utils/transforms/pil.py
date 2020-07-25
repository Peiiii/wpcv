from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import numbers, math


# Color Transforms
def _is_pil_image(img):
    return isinstance(img, Image.Image)


def rgb_to_hsv(rgb):
    import numpy as np
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv


def hsv_to_rgb(hsv):
    import numpy as np
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')


# def adjust_hue_np(img, delta):
#     import numpy as np
#     img = np.array(img)
#     hsv = rgb_to_hsv(img)
#     hsv[..., 0] += delta
#     hsv[..., 0] = hsv[..., 0] % 360
#     rgb = hsv_to_rgb(hsv)
#     img = Image.fromarray(rgb.astype(np.uint8))
#     return img
# def adjust_saturation_np(img, factor):
#     import numpy as np
#     img = np.array(img)
#     hsv = rgb_to_hsv(img)
#     hsv[..., 1] *= factor
#     hsv[..., 1] = np.clip(hsv[..., 1], 0, 1)
#     rgb = hsv_to_rgb(hsv)
#     img = Image.fromarray(rgb.astype(np.uint8))
#     return img
#
# def adjust_brightness(img,factor):
#     img=ImageEnhance.Brightness(img).enhance(factor)
#     return img
# def adjust_contrast(img,factor):
#     img=ImageEnhance.Contrast(img).enhance(factor)
#     return img
# def adjust_saturation(img,factor):
#     img=ImageEnhance.Color(img).enhance(factor)
#     return img


def adjust_sharpness(img, factor):
    img = ImageEnhance.Sharpness(img).enhance(factor)
    return img


def adjust_hsv(img, h_delta=None, s_factor=None, v_factor=None):
    import numpy as np
    '''hue:0~360,s:0~1,v:0~255'''
    rgb = np.array(img)
    if h_delta is not None:
        hsv = rgb_to_hsv(rgb)
        hsv[..., 0] += h_delta
        hsv[..., 0] = hsv[..., 0] % 360
        rgb = hsv_to_rgb(hsv)
    if s_factor is not None:
        hsv = rgb_to_hsv(rgb)
        hsv[..., 1] *= s_factor
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 1)
        rgb = hsv_to_rgb(hsv)
    if v_factor is not None:
        hsv = rgb_to_hsv(rgb)
        hsv[..., 2] *= v_factor
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)
        rgb = hsv_to_rgb(hsv)
    img = Image.fromarray(rgb.astype(np.uint8))
    return img


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image: Brightness adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image: Contrast adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image: Saturation adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image: Hue adjusted image.
    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def adjust_gamma(img, gamma, gain=1):
    r"""Perform gamma correction on an image.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}

    See `Gamma Correction`_ for more details.

    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction

    Args:
        img (PIL Image): PIL Image to be adjusted.
        gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
        gain (float): The constant multiplier.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')

    gamma_map = [255 * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

    img = img.convert(input_mode)
    return img


def to_grayscale(img, num_output_channels=1):
    """Convert image to grayscale version of image.

    Args:
        img (PIL Image): Image to be converted to grayscale.

    Returns:
        PIL Image: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel

            if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img


# Geometrical Transforms

def scale(img, scale):
    if isinstance(scale, (tuple, list)):
        scaleX, scaleY = scale
    else:
        scaleX = scaleY = scale
    w, h = img.size
    nw = int(w * scaleX)
    nh = int(h * scaleY)
    img = img.resize((nw, nh))
    return img


def resize(img, size, keep_ratio=None, fillcolor='black'):
    img = img.resize(size)
    return img


# def
def shift(img, offset, fillcolor=(0, 0, 0)):
    fill = fillcolor
    w, h = img.size
    ofx, ofy = offset
    img = ImageChops.offset(img, ofx, ofy)
    if fill is None:
        return img
    else:
        if ofx < 0:
            img.paste(fill, (ofx % w, 0, w, h))
        else:
            img.paste(fill, (0, 0, ofx, h))
        if ofy < 0:
            img.paste(fill, (0, ofy % h, w, h))
        else:
            img.paste(fill, (0, 0, w, ofy))
    return img


def translate(img, offset, fillcolor=(0, 0, 0)):
    return shift(img, offset, fillcolor)


def hflip(img):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def vflip(img):
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def horizontal_split(img, split_positions=None, split_ratios=None):
    w, h = img.size
    if split_positions is None:
        if not isinstance(split_ratios, (list, tuple)):
            split_ratios = [split_ratios]
        split_positions = [round(w * r) for r in split_ratios]
    else:
        if not isinstance(split_positions, (list, tuple)):
            split_positions = [split_positions]
    split_positions += [w]
    last_pos = 0
    res = []
    for i in range(len(split_positions)):
        pos = split_positions[i]
        l, r = last_pos, pos
        t, b = 0, h
        cropped = img.crop((l, t, r, b))
        res.append(cropped)
        last_pos = pos
    return res


def vertical_split(img, split_positions=None, split_ratios=None):
    w, h = img.size
    if split_positions is None:
        if not isinstance(split_ratios, (list, tuple)):
            split_ratios = [split_ratios]
        split_positions = [round(h * r) for r in split_ratios]
    else:
        if not isinstance(split_positions, (list, tuple)):
            split_positions = [split_positions]
    split_positions += [h]
    last_pos = 0
    res = []
    for i in range(len(split_positions)):
        pos = split_positions[i]
        t, b = last_pos, pos
        l, r = 0, w
        cropped = img.crop((l, t, r, b))
        res.append(cropped)
        last_pos = pos
    return res


def resize_to_fixed_height(img, height):
    w, h = img.size
    r = h / height
    nw = int(w / r)
    nh = int(h / r)
    img = img.resize((nw, nh))
    return img


def resize_to_fixed_width(img, width):
    w, h = img.size
    r = w / width
    nw = int(w / r)
    nh = int(h / r)
    img = img.resize((nw, nh))
    return img


def resize_by_scale(img, scale):
    w, h = img.size
    if isinstance(scale, (list, tuple)):
        rx, ry = scale
    else:
        rx = ry = scale
    nw = int(w * rx)
    nh = int(h * ry)
    img = img.resize((nw, nh))
    return img


def limit_size(img, limits):
    w, h = img.size
    mw, mh = limits
    rw = w / mw
    rh = h / mh
    r = max(rw, rh)
    if r <= 1:
        return img
    nw = int(w / r)
    nh = int(h / r)
    img = img.resize((nw, nh))
    return img


def fit_longside(img, limits):
    w, h = img.size
    mw, mh = limits
    rw = w / mw
    rh = h / mh
    r = max(rw, rh)
    nw = int(w / r)
    nh = int(h / r)
    img = img.resize((nw, nh))
    return img


def resize_keep_ratio(img, dst_size, method=Image.BICUBIC, fillcolor='black', centering=(0.5, 0.5)):
    img = ImageOps.pad(img, dst_size, method=method, color=fillcolor, centering=centering)
    return img


def pad(img, pad_ratio=None, pad_size=None, fillcolor='black'):
    w, h = img.size
    if pad_ratio is not None:
        if isinstance(pad_ratio, (tuple, list)):
            if len(pad_ratio) == 2:
                pad_l = pad_r = int(w * pad_ratio[0])
                pad_t = pad_b = int(h * pad_ratio[1])
            else:
                assert len(pad_ratio) == 4
                pad_l = int(w * pad_ratio[0])
                pad_t = int(h * pad_ratio[1])
                pad_r = int(w * pad_ratio[2])
                pad_b = int(h * pad_ratio[3])
        else:
            pad_l = pad_r = int(w * pad_ratio)
            pad_t = pad_b = int(h * pad_ratio)
    else:
        if isinstance(pad_size, (tuple, list)):
            if len(pad_size) == 2:
                pad_l = pad_r = pad_size[0]
                pad_t = pad_b = pad_size[1]
            else:
                assert len(pad_size) == 4
                pad_l, pad_t, pad_r, pad_b = pad_size
        else:
            pad_l = pad_t = pad_r = pad_b = pad_size
    dst_size = (w + pad_l + pad_r, h + pad_t + pad_b)
    canvas = Image.new(img.mode, dst_size, color=fillcolor)
    canvas.paste(img, (pad_l, pad_t))
    return canvas


def crop(img, box):
    img = img.crop(box)
    return img


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    l = int(round((h - th) / 2.))
    t = int(round((w - tw) / 2.))
    r = l + int(tw)
    b = t + int(th)
    return crop(img, [l, t, r, b])


def crop_quad(img, quad, dst_size):
    '''quad:[four points clock-wise],size:target size'''
    assert isinstance(img, Image.Image)

    def invert(points):
        points = [points[0]] + reversed(points[1:])
        return points

    quad = invert(quad)
    img = img.transform(dst_size, Image.QUAD, data=quad)
    return img


def rotate(img, degree, expand=True, fillcolor='black', translate=None):
    img = img.rotate(degree, expand=expand, fillcolor=fillcolor, translate=translate)
    return img


def shear_x(img, degree, fillcolor='black', **kwargs):
    assert isinstance(img, Image.Image)
    import math
    arc = math.radians(degree)
    factor = math.sin(arc)
    w, h = img.size
    ofx = int(h * math.sin(arc))
    dw = abs(ofx)
    img = img.transform((w + dw, h), Image.AFFINE, (1, -factor, min(0, ofx), 0, 1, 0), fill=fillcolor, **kwargs)
    return img


def shear_y(img, degree, fillcolor='black', **kwargs):
    import math
    arc = math.radians(degree)
    factor = math.sin(arc)
    w, h = img.size
    ofy = int(w * math.sin(arc))
    dh = abs(ofy)
    img = img.transform((w, h + dh), Image.AFFINE, (1, 0, 0, factor, 1, min(0, -ofy)), fill=fillcolor, **kwargs)
    return img


def shear_xy(img, degree1, degree2):
    img = shear_x(img, degree1)
    img = shear_y(img, degree2)
    return img


def shear_yx(img, degree1, degree2):
    img = shear_x(img, degree1)
    img = shear_y(img, degree2)
    return img


def perspective_transform(img, pnts1, pnts2, dst_size, resample=Image.CUBIC, fillcolor='black'):
    import numpy
    def get_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

        A = numpy.mat(matrix, dtype=numpy.float)
        B = numpy.array(pb).reshape(8)

        res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
        return numpy.array(res).reshape(8)

    assert isinstance(img, Image.Image)
    coeffs = get_coeffs(pnts2, pnts1)
    img = img.transform(dst_size, Image.PERSPECTIVE, coeffs, resample=resample, fillcolor=fillcolor)
    return img


# Image Filtering

def gaussian_blur(img, radius=2):
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


def box_blur(img, radius=3):
    img = img.filter(ImageFilter.BoxBlur(radius=radius))
    return img


def blur(img):
    img = img.filter(ImageFilter.BLUR)
    return img


def contour(img):
    img = img.filter(ImageFilter.CONTOUR)
    return img


def edge_enhance(img):
    img = img.filter(ImageFilter.EDGE_ENHANCE)
    return img


def edge_enhance_more(img):
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    return img


def emboss(img):
    img = img.filter(ImageFilter.EMBOSS)
    return img


def edge(img):
    img = img.filter(ImageFilter.FIND_EDGES)
    return img


def shapen(img):
    img = img.filter(ImageFilter.SHARPEN)
    return img


def unsharp(img, radius=5, percent=150, threshold=3):
    img = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    return img


def smooth(img):
    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img


def median_filter(img, size=5):
    img = img.filter(ImageFilter.MedianFilter(size=size))
    return img


def model_filter(img, size=5):
    img = img.filter(ImageFilter.ModeFilter(size=size))
    return img


def rank_filter(img, size=5, rank=3):
    img = img.filter(ImageFilter.RankFilter(size=size, rank=rank))
    return img


def max_filter(img, size=5):
    img = img.filter(ImageFilter.MaxFilter(size=size))
    return img


def min_filter(img, size=5):
    img = img.filter(ImageFilter.MinFilter(size=size))
    return img


def equalize(img, mask=None):
    img = ImageOps.equalize(img, mask=mask)
    return img


# Image Noise
# 高斯模糊，运动模糊，高斯白噪声，椒盐噪声，poison，

def gaussian_noise(img, var=0.1):
    import numpy as np
    img = np.array(img)
    mean = 0
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape)
    img = img + gauss
    img = Image.fromarray(img.astype(np.uint8))
    return img


def sp_noise(img, amount=0.004):
    import numpy as np
    img = np.array(img)
    s_vs_p = 0.5
    out = np.copy(img)
    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in img.shape]
    # out[coords] = 1
    out[coords] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in img.shape]
    out[coords] = 0
    img = out
    img = Image.fromarray(img.astype(np.uint8))
    return img


def poisson_noise(img):
    import numpy as np
    img = np.array(img)
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    img = np.random.poisson(img * vals) / float(vals)
    img = Image.fromarray(img.astype(np.uint8))
    return img


def speckle_noise(img):
    import numpy as np
    img = np.array(img)
    gauss = np.random.randn(*img.shape)
    img = img + img * gauss
    img = Image.fromarray(img.astype(np.uint8))
    return img


def _demo():
    from wpcv import ImageSaver
    saver = ImageSaver('/home/ars/sda5/data/tmp/0603/results', remake_dir=True)
    f = '/home/ars/图片/0.jpeg'
    # f='/home/ars/图片/2020-06-03 08-44-16 的屏幕截图.png'
    img = Image.open(f).convert('RGB')
    w, h = img.size
    print(img.size)
    assert isinstance(img, Image.Image)
    # x=Image.Image()
    # x.transform()
    # img=shift(img,(-50,-50),fill=None)
    # img=rotate(img,-90)
    # img=flip_horizontal(img)
    # img=flip_vertical(img)
    # img=shear_y(img,30)
    # img=shear_x(img,-30)
    # img=shear_xy(img,-30,30)
    # img=shear_x_rotate(img,-30,30)
    # img=img.filter(ImageFilter.BoxBlur(2))
    # im = img.filter(ImageFilter.CONTOUR)
    # saver.save(im)
    # im = img.filter(ImageFilter.DETAIL)
    # saver.save(im)
    # im = img.filter(ImageFilter.EDGE_ENHANCE)
    # saver.save(im)
    # im = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    # saver.save(im)
    # im = img.filter(ImageFilter.EMBOSS)
    # saver.save(im)
    # im = img.filter(ImageFilter.FIND_EDGES)
    # saver.save(im)
    # im = img.filter(ImageFilter.SHARPEN)
    # saver.save(im)
    # im = img.filter(ImageFilter.SMOOTH_MORE)
    # saver.save(im)
    # im = img.filter(ImageFilter.ModeFilter(size=5))
    # saver.save(im)
    # im = img.filter(ImageFilter.RankFilter(size=11, rank=3))
    # saver.save(im)
    # im = img.filter(ImageFilter.MedianFilter(size=11))
    # saver.save(im)
    # im = img.filter(ImageFilter.MinFilter(size=11))
    # saver.save(im)
    # im = img.filter(ImageFilter.UnsharpMask(11))
    # saver.save(im)
    # im = equalize(img)
    # saver.save(im)
    # im = ImageEnhance.Color(img).enhance(0.5)
    # saver.save(im)
    # # im=ImageEnhance.Brightness(img).enhance(0.2)
    # # saver.save(im)
    # im = gaussian_noise(img)
    # saver.save(im)
    # im = sp_noise(img)
    # saver.save(im)
    # im = poisson_noise(img)
    # saver.save(im)
    # im = speckle_noise(img)
    # saver.save(im)
    #
    # im = rotate(img, 30, expand=False)
    # saver.save(im)

    im = pad(img, (0.1, 0.4), fillcolor='green')
    # im = resize_keep_ratio(img, (500, 100))
    print(im.size)
    im.show()


if __name__ == '__main__':
    _demo()
