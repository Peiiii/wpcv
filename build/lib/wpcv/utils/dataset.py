import os, shutil, glob, random
# from wpkit.fsutil import copy_files_to

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

