import cv2
import numpy as np
import random,math
# 几何变换 geometrical

# 缩放
def resize(img,size):
    return cv2.resize(img,size)
def rescale(img,scale):
    if isinstance(scale,(tuple,list)):
        assert len(scale)==2
        scaleX,scaleY=scale
    else:
        scaleX=scaleY=scale
    h,w=img.shape[:2]
    h,w=int(h*scaleY),int(w*scaleX)
    return cv2.resize(img,(w,h))
# crop
def randomCrop(img,size):
    h,w=img.shape[:2]
    nw,nh=size
    spaceX=w-nw
    spaceY=h-nh
    ofx=random.randint(0,spaceX+1)
    ofy=random.randint(0,spaceY+1)
    img=img[ofy:ofy+spaceY,ofx:ofx+spaceX]
    return img
# flip
def herizontal_flip(img):
    return cv2.flip(img,1)
def vertical_flip(img):
    return cv2.flip(img,0)
# shift
def shift(image,offset):
    x,y=offset
    M = np.float32([[1, 0, x], [0, 1, y]])  # 4  //X轴移动x, Y中移动y
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))  # 5
    return shifted
# rotate

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


def rotate_box(bboxes, angle, cx, cy, h, w):
    bboxes=np.array(bboxes)
    def ltrb_to_four_corners(box):
        l, t, r, b = box
        # box2=[r,t,r,t,r,b,l,b]
        box2 = [l, t, r, t, r, b, l, b]
        return np.array(box2)

    def four_coners_to_ltrb(box, sort=False):
        x1, y1, x2, y2, x4, y4, x3, y3 = box
        l = min(x1, x3)
        r = max(x2, x4)
        t = min(y1, y2)
        b = max(y3, y4)
        if sort:
            newbox = [min(l, r), min(t, b), max(l, r), max(t, b)]
        else:
            newbox = [l, t, r, b]
        return np.array(newbox)

    corners = np.array([ltrb_to_four_corners(box) for box in bboxes])
    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)
    calculated=[four_coners_to_ltrb(box) for box in calculated]
    return calculated

def rotate_img_and_bboxes(img,bboxes,angle):
    h,w=img.shape[:2]
    img=rotate(img,angle)
    bboxes=rotate_box(bboxes,angle,w//2,h//2,h,w)
    return img,bboxes

def random_rotate_img_and_bboxes(img,bboxes,angle=30):
    if isinstance(angle,(tuple,list)):
        lower,higher=angle
    else:
        lower,higher=(-angle,angle)
    angle=random.randrange(lower,higher)
    return rotate_img_and_bboxes(img,bboxes,angle)
# affine
def affine(img,pnts1,pnts2,size=None):
    pnts1=np.array(pnts1)
    pnts2=np.array(pnts2)
    M=cv2.getAffineTransform(pnts1,pnts2)
    img=cv2.warpAffine(img,M,size)
    return img
def randomAffine(img):
    pass
# perspective
def perspective(img,pnts1,pnts2,size=None):
    pnts1=np.array(pnts1)
    pnts2=np.array(pnts2)
    M=cv2.getPerspectiveTransform(pnts1,pnts2)
    img=cv2.warpPerspective(img,M,size)
    return img
def randomPerspective(img):
    pass
# 环形图像展开成为长条形
def get_huan_by_circle(img, circle_center, radius, radius_width):
    black_img = np.zeros((radius_width, int(2 * radius * math.pi), 3), dtype='uint8')
    for row in range(0, black_img.shape[0]):
        for col in range(0, black_img.shape[1]):
            theta = math.pi * 2 / black_img.shape[1] * (col + 1)
            rho = radius - row - 1
            p_x = int(circle_center[0] + rho * math.cos(theta) + 0.5)
            p_y = int(circle_center[1] - rho * math.sin(theta) + 0.5)
            black_img[row, col, :] = img[p_y, p_x, :]
    return black_img
# distort


# 光学变换 optical
def adjust_hue(img,value):
    turn_green_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    turn_green_hsv[:, :, 0] = (turn_green_hsv[:, :, 0] + value) % 255
    turn_green_img = cv2.cvtColor(turn_green_hsv, cv2.COLOR_HSV2BGR)
    return turn_green_img
def adjust_lightness(img,r):
    darker_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    darker_hsv[:, :, 2] = r * darker_hsv[:, :, 2]
    darker_img = cv2.cvtColor(darker_hsv, cv2.COLOR_HSV2BGR)
    return darker_img

def adjust_saturation(img,r):
    colorless_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    colorless_hsv[:, :, 1] = r * colorless_hsv[:, :, 1]
    colorless_img = cv2.cvtColor(colorless_hsv, cv2.COLOR_HSV2BGR)
    return colorless_img
# gamma trans
def gamma_trans(img, gamma):
    # 感觉越大越亮
    # 具体做法是先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 实现这个映射用的是OpenCV的查表函数
    return cv2.LUT(img, gamma_table)

# PCA jitter
def PCA_Jittering(img):
    img = np.asanyarray(img, dtype='float32')
    img = img / 255.0
    img_size = img.size // 3
    img1 = img.reshape(img_size, 3)
    img1 = np.transpose(img1)
    img_cov = np.cov([img1[0], img1[1], img1[2]])
    lamda, p = np.linalg.eig(img_cov)
    p = np.transpose(p)
    alpha1 = random.gauss(0, 1)
    alpha2 = random.gauss(0, 1)
    alpha3 = random.gauss(0, 1)
    v = np.transpose((alpha1 * lamda[0], alpha2 * lamda[1], alpha3 * lamda[2]))
    add_num = np.dot(p, v)
    img2 = np.array([img[:, :, 0] + add_num[0], img[:, :, 1] + add_num[1], img[:, :, 2] + add_num[2]])
    img2 = np.swapaxes(img2, 0, 2)
    img2 = np.swapaxes(img2, 0, 1)
    return img2
# light
def equalizeHist(img):
    imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # cv2.imshow("src", img)
    channelsYUV = cv2.split(imgYUV)
    channelsYUV[0] = cv2.equalizeHist(channelsYUV[0])
    channels = cv2.merge(channelsYUV)
    result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
    return result
# adjust resolution


# blur
def motion_blur(image, degree=12, angle=45):
    image = np.array(image)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def guassion_blur(img,kernel_size=(3,3),sigmaX=0,sigmaY=0):
    img=cv2.GaussianBlur(img,kernel_size=kernel_size,sigmaX=sigmaX,sigmaY=sigmaY)
    return img

# noise
def sp_noise(image,prob):
  '''
  添加椒盐噪声
  prob:噪声比例
  '''
  output = np.zeros(image.shape,np.uint8)
  thres = 1 - prob
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      rdn = random.random()
      if rdn < prob:
        output[i][j] = 0
      elif rdn > thres:
        output[i][j] = 255
      else:
        output[i][j] = image[i][j]
  return output
def gaussion_noise(image, mean=0, var=0.001):
  '''
    添加高斯噪声
    mean : 均值
    var : 方差
  '''
  image = np.array(image/255, dtype=float)
  noise = np.random.normal(mean, var ** 0.5, image.shape)
  out = image + noise
  if out.min() < 0:
    low_clip = -1.
  else:
    low_clip = 0.
  out = np.clip(out, low_clip, 1.0)
  out = np.uint8(out*255)
  #cv.imshow("gasuss", out)
  return out
# random color

# randome **




# ##################################
def get_enclosing_box(corners):
    # corners
    """Get an enclosing box for ratated corners of a bounding box

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    Returns
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final
