import numpy as np
import math


def bounding_rect(points):
    points=np.array(points)
    l=np.min(points[:,0])
    r=np.max(points[:,0])
    t=np.min(points[:,1])
    b=np.max(points[:,1])
    return np.array([l,t,r,b])
def scale(points,scale):
    if isinstance(scale,(tuple,list)):
        scaleX,scaleY=scale
    else:
        scaleX=scaleY=scale
    points=np.array(points)*np.array([scaleX,scaleY])
    points=points.astype(np.int)
    return points
def hflip(points,width):
    points=np.array(points)
    points[:,0]=width-points[:,0]
    return points
def vflip(points,height):
    points=np.array(points)
    points[:,1]=height-points[:,1]
    return points
def translate(points,offset):
    points=np.array(points)+np.array(offset)
    return points
def shift(points,offset):
    points=np.array(points)+np.array(offset)
    return points
def get_translate_range(points,limits):
    points=np.array(points,dtype=np.int).reshape(-1,2)
    if len(limits)==2:
        ml=mt=0
        mr,mb=limits
    else:
        ml,mt,mr,mb=limits
    l,t,r,b=bounding_rect(points)
    # xrange,yrange=[ml-l,mr-r],[mt-t,mb-b]
    range=[ml-l,mt-t,mr-r,mb-b]
    return range
    # return xrange,yrange

def rotate(points,degree,center,img_size=None,expand=None):
    angle=math.radians(degree)
    M=[
        [math.cos(angle),math.sin(angle)],
        [-math.sin(angle),math.cos(angle)]
    ]
    M=np.array(M)
    points=np.array(points)
    center=np.mat(center)
    def _rotate_points(points,M,center):
        points=M.dot((points-center).T)+center.T
        points=np.array(points.T)
        return points
    points=_rotate_points(points,M,center)
    if expand:
        def _get_img_boundary(img_size):
            w,h=img_size
            bnd=np.array([[0,0],[w,0],[0,h],[w,h]])
            bnd=_rotate_points(bnd,M,center)
            bnd=bounding_rect(bnd)
            return bnd
        l,t,r,b=_get_img_boundary(img_size)
        points=points-np.array([l,t])
    points=np.array(points).astype(np.int).reshape((-1,2))
    return points

def shear_x(points,degree,img_size=None,expand=None):
    '''[x2,y2].T=[[1,sin],[0,cos]]*[x1,y1].T

    '''
    angle=math.radians(degree)
    M=np.array([
        [1,math.sin(angle)],
        [0,1]
    ])
    points=np.mat(points)*M.T
    if expand:
        def _get_img_boundary(img_size):
            w,h=img_size
            bnd=np.mat([[0,0],[w,0],[0,h],[w,h]])
            bnd=bnd*M.T
            bnd=bounding_rect(bnd)
            return bnd

        l,t,r,b=_get_img_boundary(img_size)
        points = points - np.array([l, t])
    points = np.array(points).astype(np.int).reshape((-1, 2))
    return points

def shear_y(points,degree,img_size=None,expand=None):
    '''[x2,y2].T=[[1,sin],[0,cos]]*[x1,y1].T

    '''
    angle=math.radians(degree)
    M=np.array([
        [1,0],
        [-math.sin(angle),1]
    ])
    points=np.mat(points)*M.T
    # expand=None
    if expand:
        def _get_img_boundary(img_size):
            w,h=img_size
            bnd=np.mat([[0,0],[w,0],[0,h],[w,h]])
            bnd=bnd*M.T
            bnd=bounding_rect(bnd)
            return bnd

        l,t,r,b=_get_img_boundary(img_size)
        points = points - np.array([l, t])
    points = np.array(points).astype(np.int).reshape((-1, 2))
    return points
