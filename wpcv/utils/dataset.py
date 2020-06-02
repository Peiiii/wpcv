import os, shutil, glob, random,time
def newdir(out_dir):
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    os.makedirs(out_dir)
def copy_files_to(files,dst,overwrite=False):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for i,f in enumerate(files):
        fn=os.path.basename(f)
        f2=dst+'/'+fn
        if os.path.exists(f2):
            if os.path.samefile(f, f2):
                print("ignoring same file:", f, f2)
                continue
            if not overwrite:
                raise Exception("file %s already exists."%(f2))
            else:
                print("overwriting %s to %s ..."%(f,f2))
                os.remove(f2)
        shutil.copy(f,f2)
def split_train_val(data_dir, train_dir, val_dir, val_split=0.1, num_val=None, ext='.jpg', shuffle=True, sort=False):
    newdir(train_dir)
    newdir(val_dir)
    fs = glob.glob(data_dir + '/*' + ext)
    if sort:
        fs.sort()
    elif shuffle:
        random.shuffle(fs)
    if not num_val:
        num_val = int(len(fs) * val_split)
    val_files = fs[:num_val]
    train_files = fs[num_val:]
    copy_files_to(train_files, train_dir)
    copy_files_to(val_files, val_dir)

def split_list(lis,ratios,shuffle=True):
    if not lis:
        return None
    if len(lis)<len(ratios):
        raise Exception("List is not long enough to be split.")
    import numpy as np
    if shuffle:
        random.shuffle(lis)
    ratios=np.array(ratios)
    ratios=ratios/ratios.sum()
    nums=ratios*len(lis)
    nums=np.round(nums).astype(int)
    total=len(lis)
    splits=[]
    current_index=0
    for i,num in enumerate(nums):
        end_point=min(current_index+num,total)
        batch=lis[current_index:end_point]
        splits.append(batch)
        current_index=end_point
    return splits


def split_files(src_files,dst_dirs_and_ratios:'{"output":{"train":0.8,"val":0.2}}',shuffle=True,remake_dirs=True):
    def compress_tree(tree,root=''):
        leaves = {}
        def parse_tree(root, dic):
            for k, v in dic.items():
                p = os.path.join(root,k)
                if isinstance(v, dict):
                    parse_tree(p, v)
                else:
                    '''Now it should be a number'''
                    leaves[p] = v
        parse_tree(root=root,dic=tree)
        return leaves
    tree=compress_tree(dst_dirs_and_ratios)
    out_dirs=list(tree.keys())
    ratios=list(tree.values())
    out_file_lists=split_list(src_files,ratios,shuffle=shuffle)
    for dir,files in zip(out_dirs,out_file_lists):
        if remake_dirs and os.path.exists(dir):
            shutil.rmtree(dir)
            time.sleep(0.01)
        print('Coping %s files to %s : %s'%(len(files),dir,files))
        copy_files_to(files,dir)

def split_dir(src_dir,dst_dirs_and_ratios:'{"output":{"train":0.8,"val":0.2}}',shuffle=True,remake_dirs=True,glob_strings=['*.jpg','*.png']):
    fs=[]
    for string in glob_strings:
        fs+=glob.glob(os.path.join(src_dir,string))
    split_files(
        src_files=fs,dst_dirs_and_ratios=dst_dirs_and_ratios,shuffle=shuffle,remake_dirs=remake_dirs
    )
    print('Split dir %s finished.'%(src_dir))



def split_train_val_imagefolder(data_dir, train_dir, val_dir, val_split=0.1, num_val_cls=None, ext='.jpg', shuffle=True,
                                sort=False):
    newdir(train_dir)
    newdir(val_dir)
    for cls in os.listdir(data_dir):
        cls_dir = data_dir + '/' + cls
        train_cls_dir = train_dir + '/' + cls
        val_cls_dir = val_dir + '/' + cls
        split_train_val(cls_dir, train_dir=train_cls_dir, val_dir=val_cls_dir, val_split=val_split, num_val=num_val_cls,
                        ext=ext, shuffle=shuffle, sort=sort)
def merge_dirs(src_dirs,dst_dir):
    if  os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
        a=0
    os.makedirs(dst_dir)
    for dir in src_dirs:
        fs=glob.glob(dir+'/*')
        copy_files_to(fs,dst_dir)
if __name__ == '__main__':
    split_train_val_imagefolder(
        data_dir='/home/ars/disk/datasets/cifar-10-python/cofar10-imagenet-format',
        train_dir='/home/ars/disk/datasets/cifar-10-python/cofar10-trainval/train/raw',
        val_dir='/home/ars/disk/datasets/cifar-10-python/cofar10-trainval/val',
        # val_split=0.1
        num_val_cls=500
    )

    split_train_val_imagefolder(
        data_dir='/home/ars/disk/datasets/cifar-10-python/cofar10-trainval/train/raw',
        train_dir='/home/ars/disk/datasets/cifar-10-python/cofar10-trainval/train/unlabeled_unmerged',
        val_dir='/home/ars/disk/datasets/cifar-10-python/cofar10-trainval/train/labeled',
        # val_split=0.1
        num_val_cls=500
    )

    merge_dirs(glob.glob('/home/ars/disk/datasets/cifar-10-python/cofar10-trainval/train/unlabeled_unmerged'+'/*'),
               dst_dir='/home/ars/disk/datasets/cifar-10-python/cofar10-trainval/train/unlabeled')

