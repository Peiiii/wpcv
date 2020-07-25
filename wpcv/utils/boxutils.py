import math
##########################Box Operations#############################
def resize_box(box,size):
    cx,cy,w,h=ltrb_to_ccwh(box)
    nw,nh=size
    l=cx-nw//2
    r=cx+nw//2
    t=cy-nh//2
    b=cy+nh//2
    return [l,t,r,b]
def rescale_box(box,scale):
    '''rescale with  image'''
    if isinstance(scale,(tuple,list)):
        scaleX,scaleY=scale
    else:
        scaleX=scaleY=scale
    l,t,r,b=box
    l*=scaleX
    r*=scaleX
    t*=scaleY
    b*=scaleY
    return [l,t,r,b]
def get_rotate_info(angle,cx,cy,h,w):
    import math
    import numpy as np
    angle = math.radians(angle)
    cos = math.cos(angle)
    sin = math.sin(angle)
    M = np.array([[cos, sin], [-sin, cos]])
    cos = abs(cos)
    sin = abs(sin)
    nw = w * cos + h * sin
    nh = w * sin + h * cos
    return M,nw,nh
def rotate_points(points,angle,cx,cy,h,w):
    import numpy as np
    M,nw,nh=get_rotate_info(angle,cx,cy,h,w)
    points=np.array(points)
    original_shape=points.shape
    points=points.reshape((-1,2))
    points=M.dot((points-np.array([cx,cy])).T).T+np.array([nw//2,nh//2])
    points=points.reshape(original_shape)
    return points

def rotate_boxes(boxes,angle,cx,cy,h,w):
    '''原创'''
    boxes=[list(box) for box in boxes]
    corners = [ltrb_to_four_corners(box) for box in boxes]
    corners=rotate_points(corners,angle,cx,cy,h,w)
    boxes=[four_coners_to_ltrb(quad_to_four_corners(organize_quad_points(four_corners_to_quad(list(box))))) for box in corners]
    return boxes
def rotate_box(box, angle, cx, cy, h, w):
    return rotate_boxes([box],angle,cx,cy,h,w)
def shift_box(box,offset,limits=None):
    l,t,r,b=box
    ofx,ofy=offset
    l+=ofx
    r+=ofx
    t+=ofy
    b+=ofy
    return limit_box([l,t,r,b],limits=limits)
def translate_box(box,offset,limits=None):
    return shift_box(box,offset,limits)
def flip_box_horizontal(box,imsize):
    imw,imh=imsize
    l,t,r,b=box
    l=imw-r
    r=imw-l
    return [l,t,r,b]
def flip_box_vertical(box,imsize):
    imw,imh=imsize
    l,t,r,b=box
    t=imh-b
    b=imh-t
    return [l,t,r,b]
def pad_box(box,pad_size=5,pad_ratio=None,limits=None):
    from math import inf
    limits=limits or (-inf,-inf,inf,inf)
    l,t,r,b=box
    bh=b-t
    bw=r-l
    if pad_ratio is not None:
        if isinstance(pad_ratio,(tuple,list)):
            if len(pad_ratio)==2:
                pad_l=pad_r=int(min(bh,bw)*pad_ratio[0])
                pad_t=pad_b=int(min(bh,bw)*pad_ratio[1])
            else:
                assert len(pad_ratio)==4
                pad_l= int(min(bh, bw) * pad_ratio[0])
                pad_t= int(min(bh, bw) * pad_ratio[1])
                pad_r= int(min(bh, bw) * pad_ratio[2])
                pad_b= int(min(bh, bw) * pad_ratio[3])
        else:
            pad_l=pad_t=pad_r=pad_b=int(min(bh,bw)*pad_ratio)
    else:
        if isinstance(pad_size,(tuple,list)):
            if len(pad_size)==2:
                pad_l=pad_r=pad_size[0]
                pad_t=pad_b=pad_size[1]
            else:
                assert len(pad_size)==4
                pad_l,pad_t,pad_r,pad_b=pad_size
        else:
            pad_l=pad_t=pad_r=pad_b=pad_size
    l-=pad_l
    t-=pad_t
    r+=pad_r
    b+=pad_b
    return limit_box([l,t,r,b],limits=limits)

def pad_quad(quad,pad_size=5,pad_ratio=None,limits=None):
    import math
    from math import inf
    limits=limits or (-inf,-inf,inf,inf)
    def dist(p1,p2):
        return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    def mean(li):
        return sum(li)/len(li)
    def minus(la,lb):
        return [x-y for x,y in zip(la,lb)]
    def add(la,lb):
        return [x +y for x, y in zip(la, lb)]
    def get_pad_params(pad_size,pad_ratio,quad ):
        # l, t, r, b = box
        p0,p1,p2,p3=quad
        bh = mean([dist(p0,p3),dist(p1,p2)])
        bw = mean([dist(p0,p1),dist(p3,p2)])
        if pad_ratio is not None:
            if isinstance(pad_ratio, (tuple, list)):
                if len(pad_ratio) == 2:
                    pad_l = pad_r = int(min(bh, bw) * pad_ratio[0])
                    pad_t = pad_b = int(min(bh, bw) * pad_ratio[1])
                else:
                    assert len(pad_ratio) == 4
                    pad_l = int(min(bh, bw) * pad_ratio[0])
                    pad_t = int(min(bh, bw) * pad_ratio[1])
                    pad_r = int(min(bh, bw) * pad_ratio[2])
                    pad_b = int(min(bh, bw) * pad_ratio[3])
            else:
                pad_l = pad_t = pad_r = pad_b = int(min(bh, bw) * pad_ratio)
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
        return pad_l,pad_t,pad_r,pad_b
    pad_l, pad_t, pad_r, pad_b=get_pad_params(pad_size,pad_ratio,quad)
    p0, p1, p2, p3 = quad
    p0=add(p0,[-pad_l,-pad_t])
    p1=add(p1,[pad_r,-pad_t])
    p2=add(p2,[pad_r,pad_b])
    p3=add(p3,[-pad_l,pad_b])

    return limit_quad([p0,p1,p2,p3],limits=limits)

def limit_point(p,limits):
    if limits is None:return p
    if len(limits)==2:
        ml,mt=0,0
        mr,mb=limits
    else:
        assert len(limits)==4
        ml,mt,mr,mb=limits
    x,y=p
    x=max(ml,min(mr,x))
    y=max(mt,min(mb,y))
    return [x,y]

def limit_quad(quad,limits=None):
    quad=[limit_point(p,limits) for p in quad]
    return quad

def limit_box(box,limits=None):
    if limits is None:return box
    if len(limits)==2:
        ml,mt=0,0
        mr,mb=limits
    else:
        assert len(limits)==4
        ml,mt,mr,mb=limits
    l,t,r,b=box
    l=max(ml,l)
    t=max(mt,t)
    r=min(mr,r)
    b=min(mb,b)
    if l>=r:
        return None
    if t>=b:return None
    return [l,t,r,b]
############################Sort Operations#############################
def min_enclosing_box(boxes):
    assert len(boxes)
    ml,mt,mr,mb=boxes[0]
    for box in boxes[1:]:
        l,t,r,b=box
        ml=min(ml,l)
        mt=min(mt,t)
        mr=max(mr,r)
        mb=max(mb,b)
    return [ml,mt,mr,mb]
def box_in_area(box,size):
    l,t,r,b=box
    mr,mb=size
    if l<0 or t<0 or r>mr or b>mb:
        return False
    else:
        return True

def organize_quad_points(box):
    '''
    turn into clock-wise points: quad
    p0:lt,p1:rt,p2:rb,p4:lb
    '''
    p0,p3, p1,p2 = sorted(box, key=lambda p: p[0])
    p03=[p0,p3]
    p12=[p1,p2]
    p0, p3 = sorted(p03, key=lambda p: p[1])
    p1, p2 = sorted(p12, key=lambda p: p[1])
    return [p0,p1,p2,p3]
def sort_boxes(boxes):
    from functools import cmp_to_key
    boxes=[ltrb_to_ccwh(box) for box in boxes]
    boxes=[[box] for box in boxes]
    def block_cmp(b1, b2):
        list1 = [b1[0][0], b1[0][1], b1[0][2], b1[0][3]]
        list2 = [b2[0][0], b2[0][1], b2[0][2], b2[0][3]]
        if len(b1[0]) == 4:
            list1[0] += list1[2] / 2
            list1[1] += list1[3] / 2
            list2[0] += list2[2] / 2
            list2[1] += list2[3] / 2

        flag = 1
        if list1[0] > list2[0]:
            list1, list2 = list2, list1
            flag = -1

        if list2[1] + list1[3] / 2 < list1[1]:
            return flag
        return -flag

    boxes.sort(key=cmp_to_key(block_cmp), reverse=False)
    boxes=[box[0] for box in boxes]
    boxes=[ccwh_to_ltrb(box) for box in boxes]
    return boxes
############################ConvertingTools#############################
def ltrb_to_four_corners(box):
    '''four corners: 8-d vector clock-wise'''
    l,t,r,b=box
    box2=[l,t,r,t,r,b,l,b]
    return box2
def four_coners_to_ltrb(box,sort=False):
    x1,y1,x2,y2,x4,y4,x3,y3=box
    l=min(x1,x3)
    r=max(x2,x4)
    t=min(y1,y2)
    b=max(y3,y4)
    if sort:
        newbox = [min(l, r), min(t, b), max(l, r), max(t, b)]
    else:
        newbox=[l,t,r,b]
    return newbox
def ltrb_to_ccwh(box):
    l, t, r, b = box
    w = r - l
    h = b - t
    cx = (l + r) // 2
    cy = (t + b) // 2
    return [cx,cy,w,h]
def ccwh_to_ltrb(box):
    cx,cy,w,h=box
    l=cx-w//2
    t=cy-h//2
    r=cx+w//2
    b=cy+h//2
    return [l,t,r,b]
def oowh_to_ltrb(box):
    '''o:origin'''
    x, y, w, h = box
    return [int(i) for i in (x, y, x + w, y + h)]
def ltrb_to_oowh(box):
    l, t, r, b = box
    return [l, t, r - l, b - t]
def xywh_to_ltrb(box):
    '''same with oowh'''
    x,y,w,h=box
    return [int(i) for i in (x,y,x+w,y+h)]
def ltrb_to_xywh(box):
    l,t,r,b=box
    return [l,t,r-l,b-t]
def quad_to_ltrb(box):
    '''quad: clock-wise four points'''
    (x0,y0),(x1,y1),(x2,y2),(x3,y3)=box
    return [min(x0,x3),min(y0,y1),max(x1,x2),max(y2,y3)]
def ltrb_to_quad(box):
    x0,y0,x1,y1=box
    return [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
def quad_to_four_corners(quad):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad
    return [x0,y0,x1,y1,x2,y2,x3,y3]
def four_corners_to_quad(box):
    x0,y0,x1,y1,x2,y2,x3,y3 = box
    return [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
###########################Calculations##############################
def calc_iou(box1,box2):
    l1,t1,r1,b1=box1
    l2,t2,r2,b2=box2
    w1,h1=r1-l1,b1-t1
    w2,h2=r2-l2,b2-t2
    width=min(r1,r2)-max(l1,l2)
    height=min(b1,b2)-max(t1,t2)
    width=max(width,0)
    height=max(height,0)
    area=width*height
    return area/(w1*h1+w2*h2-area)
def calc_iou2(box1,box2):
    width=min(box1[2],box2[2])-max(box1[0],box2[0])
    height=min(box1[3],box2[3])-max(box1[1],box2[1])
    width=max(width,0)
    height=max(height,0)
    area=width*height
    return area/((box1[2]-box1[0])*(box1[3]-box1[1])+(box2[2]-box2[0])*(box2[3]-box2[1])-area)
def calc_iou_batch_with_batch(boxes1,boxes2):
    box1=boxes1
    box2=boxes2
    import numpy as np
    box1=np.array(box1)
    box2=np.array(box2)
    width = np.min(np.vstack([box1[...,2], box2[...,2]]),axis=0) - np.max(np.vstack([box1[...,0], box2[...,0]]),axis=0)
    height = np.min(np.vstack([box1[...,3], box2[...,3]]),axis=0) - np.max(np.vstack([box1[...,1], box2[...,1]]),axis=0)
    width = np.clip(width,0,np.inf)
    height = np.clip(height,0,np.inf)
    area = width * height
    return area / ((box1[...,2] - box1[...,0]) * (box1[...,3] - box1[...,1]) + (box2[...,2] - box2[...,0]) * (box2[...,3] - box2[...,1]) - area)
def calc_iou_batch(boxes,box):
    '''calculate iou between batch bboxes with one bbox , 原创'''
    box1=boxes
    box2=box
    import numpy as np
    box1 = np.array(box1)
    box2 = np.array(box2)
    shape=box1.shape
    width = np.min(np.vstack([box1[..., 2], np.full(shape[:-1],box2[2])]), axis=0) - np.max(np.vstack([box1[..., 0], np.full(shape[:-1],box2[0])]),axis=0)
    height = np.min(np.vstack([box1[..., 3], np.full(shape[:-1],box2[3])]), axis=0) - np.max(np.vstack([box1[..., 1], np.full(shape[:-1],box2[1])]),axis=0)
    width = np.clip(width, 0, np.inf)
    height = np.clip(height, 0, np.inf)
    area = width * height
    return area / ((box1[...,2] - box1[...,0]) * (box1[...,3] - box1[...,1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - area)
def calc_quad_angle(quad):
    '''cal angle of a quad'''
    p0,p1,p2,p3=quad
    def center(p1,p2):
        x1,y1=p1
        x2,y2=p2
        return (x1+x2)/2,(y1+y2)/2
    x1,y1=center(p0,p3)
    x2,y2=center(p1,p2)
    angle=math.atan2(y1-y2,x2-x1)*180/math.pi
    return angle
def calc_box_area(box):
    l,t,r,b=box
    return (r-l)*(b-t)
def calc_polygon_area(points):
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return abs(area / 2)
##############################TextBox################################
class TextBox(dict):
    def __init__(self):
        super().__init__()
    def parse(self):
        p0,p1,p2,p3=quad=self['quad']
        xmin,ymin,xmax,ymax=l,t,r,b=ltrb=quad_to_ltrb(quad)
        ox,oy,w,h=oowh=ltrb_to_oowh(ltrb)
        cx,cy,w,h=ccwh=ltrb_to_ccwh(ltrb)
        angle=calc_quad_angle(quad)
        text=None
        prob=None
        self.update(
            ltrb=ltrb,oowh=oowh,ccwh=ccwh,
            xmin=xmin,ymin=ymin,xmax=xmax,ymax=ymax,
            l=l,t=t,r=r,b=b,
            ox=ox,oy=oy,w=w,h=h,
            cx=cx,cy=cy,
            angle=angle,
            text=text,
            prob=prob
        )
        return self
