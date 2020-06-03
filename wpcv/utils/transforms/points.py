import numpy as np

def bounding_rect(points):
    points=np.array(points)
    l=np.min(points[:,0])
    r=np.max(points[:,0])
    t=np.min(points[:,1])
    b=np.min(points[:,1])
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
def get_translate_range(points,limits):
    if len(limits)==2:
        ml=mt=0
        mr,mb=limits
    else:
        ml,mt,mr,mb=limits
    l,t,r,b=bounding_rect(points)
    xrange,yrange=[l-ml,mr-r],[t-mt,mb-b]
    return xrange,yrange

# def shear():
#     x2-x1=y1*sin
#     y2=y1*cos
#     [x2,y2]=[x1,y1]*[[1,sin],[0,cos]]
# def rotate():
#     [x2,y2]-[c2x,c2y]=[[cos,-sin],[sin,cos]]*([x1,y1]-[c1x,c1y])
#     [x2,y2]=M*[x1,y1]-M*[c1x,c1y]+[c2x,c2y]


# def shift(points,offset):
