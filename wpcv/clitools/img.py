import os,shutil,glob
from PIL import Image
import cv2

def convert_image_suffix_in_dir(ext1,ext2,dir=None,dst_dir=None,copy=False,verbose=False):
	if not dir:
		dir=os.getcwd()
	if not dst_dir:
		dst_dir=dir
	print('input-dir:%s ,output-dir:%s'%(dir,dst_dir))
	fs=glob.glob(dir+'/*'+ext1)
	for i,f in enumerate(fs):
		img=cv2.imread(f)
		f2=dst_dir+'/'+os.path.basename(f).replace(ext1,ext2)
		cv2.imwrite(f2,img)
		if not copy:
			os.remove(f)
		if verbose:
			print(i,f,f2)


