from .img import *
import fire

class CLI:
	def cvt_img_dir(self,ext1,ext2,input=None,out=None,copy=False,verbose=False):
		return convert_image_suffix_in_dir(ext1=ext1,ext2=ext2,dir=input,dst_dir=out,copy=copy,verbose=verbose)

def main():
	cli=CLI()
	fire.Fire(cli)
