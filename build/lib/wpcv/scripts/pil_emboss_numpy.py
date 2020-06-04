from PIL import Image
import numpy

# defining azimuth, elevation, and depth
ele = numpy.pi/2.2 # radians
azi = numpy.pi/4.  # radians
dep = 10.          # (0-100)

# get a B&W version of the image
img = Image.open('/home/ars/图片/2019011816055827.jpg').convert('L')
# get an array
a = numpy.asarray(img).astype('float')
# find the gradient
grad = numpy.gradient(a)
# (it is two arrays: grad_x and grad_y)
grad_x, grad_y = grad
# getting the unit incident ray
gd = numpy.cos(ele) # length of projection of ray on ground plane
dx = gd*numpy.cos(azi)
dy = gd*numpy.sin(azi)
dz = numpy.sin(ele)
# adjusting the gradient by the "depth" factor
# (I think this is how GIMP defines it)
grad_x = grad_x*dep/100.
grad_y = grad_y*dep/100.
# finding the unit normal vectors for the image
leng = numpy.sqrt(grad_x**2 + grad_y**2 + 1.)
uni_x = grad_x/leng
uni_y = grad_y/leng
uni_z = 1./leng
# take the dot product
a2 = 255*(dx*uni_x + dy*uni_y + dz*uni_z)
# avoid overflow
a2 = a2.clip(0,255)
# you must convert back to uint8 /before/ converting to an image
img2 = Image.fromarray(a2.astype('uint8'))
img2.save('daisy2.png')