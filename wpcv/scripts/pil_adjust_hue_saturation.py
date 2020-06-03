from PIL import Image
import numpy as np


def rgb_to_hsv(rgb):
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


def adjust_hue(img, delta):
    img = np.array(img)
    hsv = rgb_to_hsv(img)
    hsv[..., 0] += delta
    hsv[..., 0] = hsv[..., 0] % 360
    rgb = hsv_to_rgb(hsv)
    img = Image.fromarray(rgb.astype(np.uint8))
    return img


def adjust_saturation(img, factor):
    img = np.array(img)
    hsv = rgb_to_hsv(img)
    hsv[..., 1] *= factor
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 1)
    rgb = hsv_to_rgb(hsv)
    img = Image.fromarray(rgb.astype(np.uint8))
    return img


def adjust_hsv(img, h_delta=None, s_factor=None, v_factor=None):
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


if __name__ == '__main__':
    green_hue = (180 - 78) / 360.0
    red_hue = (180 - 180) / 360.0

    img = Image.open('/home/ars/图片/0.jpg')
    # img=adjust_hue(img,100)
    # img=adjust_saturation(img,0.02)
    img = adjust_hsv(img, 100, 0.98, 0.7)
    img.show()
