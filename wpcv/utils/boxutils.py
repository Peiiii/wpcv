import cv2
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from matplotlib import pyplot as plt
import os,shutil,glob

#########################Fonts###########################
default_font=None
default_font_path=None
def get_default_font(font_size=None):
    return ImageFont.truetype(default_font_path,font_size)
def set_font_path(path):
    global default_font_path
    default_font_path=path
def set_font(path,size=32):
    global default_font
    default_font=ImageFont.truetype(path,size=size)
def test_font_dir(dir,dst=None,text=None):
    fs=glob.glob(dir+'/*.ttf')+glob.glob(dir+'/*.ttc')+glob.glob(dir+'/*.otf')
    dst=dst or dir+'/test_font'
    for i,f in enumerate(fs):
        test_font(f,dst,text=text)
        print(i,f)
    print('finished.')
def test_font(path,dst='./test_font',text=None):
    img_dir=dst+'/imgs'
    log_file=dst+'/font_errors.txt'
    bad_fonts=dst+'/bad_fonts.txt'
    img=blank_rgb(size=(1024,32))
    font=ImageFont.truetype(path,size=24)
    text=text or 'Hello! 今天过得怎么样,~!#$%^&*()_+=-'
    try:
        img=draw_text(img,text=text,font=font)
    except:
        msg='Error occured when handle %s'%(path)
        print(msg)
        with open(log_file,'a',encoding='utf-8') as f:
            f.write(msg+'\n')
        with open(bad_fonts,'a',encoding='utf-8') as f:
            f.write(path+'\n')
        return
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    img.save(img_dir+'/'+os.path.basename(path)[:-3]+'.jpg')
####################Generate#####################
def blank_rgb(size=(512,48),color='white'):
    img=Image.new('RGB',size,color)
    return img
def new_blank_img_as(img):
    img=Image.new('RGB',size=img.size,color='white')
    return img
###################Size#######################
def resize_to_fixed_height(img,height):
    w,h=img.size
    r=h/height
    nw=int(w/r)
    nh=int(h/r)
    img=img.resize((nw,nh))
    return img
def resize_to_fixed_width(img,width):
    w,h=img.size
    r=w/width
    nw=int(w/r)
    nh=int(h/r)
    img=img.resize((nw,nh))
    return img
def resize_by_scale(img,scale):
    w, h = img.size
    r = scale
    nw = int(w * r)
    nh = int(h * r)
    img = img.resize((nw, nh))
    return img
def limit_size(img,limits):
    w,h=img.size
    mw,mh=limits
    rw=w/mw
    rh=h/mh
    r=max(rw,rh)
    if r<=1:
        return img
    nw=int(w/r)
    nh=int(h/r)
    img=img.resize((nw,nh))
    return img
#####################Draw#########################
def concat_imgs_horizontal(imgs):
    imgs=[cv2img(img) for img in imgs]
    img=np.concatenate(imgs,axis=1)
    return pilimg(img)
def concat_imgs_vertical(imgs):
    imgs=[cv2img(img) for img in imgs]
    img=np.concatenate(imgs,axis=0)
    return pilimg(img)
def draw_boxes_with_label(img,boxes,offset=(0,-16),box_color='red',text_color='green',line_width=5):
    ofx,ofy=offset
    for box,label in boxes:
        l,t,r,b=box
        img=draw_box(img,box,outline=box_color,width=line_width)
        img=draw_text(img,text=label,xy=(l+ofx,t+ofy),fill=text_color)
    return img
def mark_img(img,text,font_path=None,font_size=None):
    imw,imh=img.size
    font_path=font_path or default_font_path
    if font_path:
        font_size=font_size or min(32,imw//len(text))
        font=ImageFont.truetype(font_path,size=font_size)
    else:
        font=None
    def calc_pos(imsize,boxsize):
        imw,imh=imsize
        w,h=boxsize
        x=imw//2-w//2
        y=imh//2-h//2
        return (x,y)

    xy=calc_pos(img.size,(font.size*len(text),font.size))
    img=draw_text(img,text,xy=xy,fill='red',font=font)
    return img
def draw_text(img,text,xy=(0,0),fill='black',font=None):
    font=font or default_font
    draw=ImageDraw.ImageDraw(img)
    draw.text(xy,text=text,fill=fill,font=font)
    return img
def draw_boxes(img,boxes,copy=False,*args,**kwargs):
    if copy:
        img=img.copy()
    for box in boxes:img=draw_box(img,box,copy=False,*args,**kwargs)
    return img
def draw_box(img,box,copy=True,width=5,outline='red',fill=None):
    box=tuple(box)
    if copy:
        img=img.copy()
    draw=ImageDraw.Draw(img)
    draw.rectangle((box[:2],box[2:]),width=width,outline=outline,fill=fill)
    return img
def draw_polygon(img,points,color='blue',width=1):
    points=[tuple(p) for p in points]
    draw=ImageDraw.Draw(img)
    points.append(points[0])
    draw.line(points,fill=color,width=width)
    return img
def draw_textboxes(img,textboxes,copy=False,font_size=32):
    if copy:
        img=img.copy()
    for textbox in textboxes:
        img=draw_textbox(img,textbox,copy=False,font_size=font_size)
    return img
def draw_textbox(img,textbox,copy=True,font_size=None):
    import os
    font_size=font_size or 32
    box,text=textbox
    if copy:
        img=img.copy()
    draw = ImageDraw.Draw(img)
    font=ImageFont.truetype(os.path.dirname(__file__)+'/msyh.ttf',size=font_size)
    draw.text(box[:2],text=text,fill='black',font=font)
    return img
def pad_text_right(img,text,pad_width=400,pad_ratio=1,font=None,font_size=None,line_length=None):
    # img=cv2img(img)
    # h,w=img.shape[:2]
    # blank=np.zeros_like(img)
    def get_max_line_width(text):
        lines=text.split('\n')
        lens=[len(line) for line in lines]+[20]
        return max(*lens)
    line_length=line_length or max(get_max_line_width(text),20)
    w,h=img.size
    if pad_width:
        pad_width=int(w*pad_ratio)
    bw=pad_width
    canvas = Image.new(img.mode, (w+bw,h),'white')
    blank=Image.new(img.mode,(bw,h),color='white')
    font_size=font_size or max(16,int(bw/line_length))
    default_font = get_default_font(font_size=font_size)
    font = font or default_font
    blank=draw_text(blank,text,(5,5),font=font)
    canvas.paste(img)
    canvas.paste(blank,(w,0,w+bw,h))
    return canvas
########################Crop########################
def crop_boxes(img,boxes):
    imgs=[]
    for box in boxes:
        im=crop(img,box)
        imgs.append(im)
    return imgs
def iter_boxes(img,boxes,do):
    reses=[]
    for box in boxes:
        im=crop(img,box)
        res=do(im)
        reses.append(res)
    return reses
def crop(img,bbox):
    return img.crop(bbox)
def crop_by_ratio(img,rbox):
    w,h=img.size
    box=tuple([int(x) for x in (rbox[0]*w,rbox[1]*h,rbox[2]*w,rbox[3]*h)])
    return img.crop(box)
def crop_quad(img,box):
    p0, p1, p2, p3 = box
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = box
    w,h=((x1-x0+x2-x3)//2,(y3-y0+y2-y1)//2)
    w,h=int(w),int(h)
    M=cv2.getPerspectiveTransform(np.float32([p0,p1,p3,p2]),np.float32([[0,0],[w,0],[0,h],[w,h]]))
    img=cv2.warpPerspective(img,M,(w,h))
    return img
def crop_quads(img,boxes):
    bims=[]
    for box in boxes:
        bims.append(crop_quad(img,box))
    return bims
#######################Show#############################
def cv2img(img):
    if isinstance(img,Image.Image):
        img=np.array(img)
        if len(img.shape)==3:img=img[:,:,::-1]
        return img
    return img
def pilimg(img):
    if isinstance(img,Image.Image):return img
    if isinstance(img,np.ndarray):
        if len(img.shape)==3:img=img[:,:,::-1]
    return Image.fromarray(np.array(img).astype(np.uint8))
def pilimshow(x,*args,**kwargs):
    x=pilimg(x)
    x.show(*args,**kwargs)
def cv2imshow(x,title='cv2 image'):
    x=cv2img(x)
    cv2.imshow(title,x)
    cv2.waitKey(0)
def pltimshow(x,*args,**kwargs):
    x=cv2img(x)
    plt.imshow(x,*args,**kwargs)
    plt.show()
########################ImageSaver###########################
class ImageSaver:
    def __init__(self,save_dir=None,remake_dir=False,start_file_index=0):
        save_dir=save_dir or './tmp_image_save_dir'
        import time
        if os.path.exists(save_dir) and remake_dir:
            shutil.rmtree(save_dir)
            time.sleep(0.01)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir=save_dir
        self.remake_dir=remake_dir
        self.file_index=start_file_index
        self.alive=True
    def deactive(self):
        self.alive=False
    def active(self):
        self.alive=True
    def save(self,img,name=None):
        if not self.alive:
            return False
        name=name or '%s.jpg'%self.file_index
        name=os.path.join(self.save_dir,name)
        if isinstance(img, Image.Image):
            img.save(name)
        else:
            cv2.imencode('.jpg', img)[1].tofile(name)
        self.file_index+=1
        return name
