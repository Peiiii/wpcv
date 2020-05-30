import json
import os
import cv2,shutil,glob
import random

random.seed(0)

# 根路径，里面包含images(图片文件夹)，annos.txt(bbox标注)，classes.txt(类别标签),以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
root_path = '/home/ars/sda5/datasets/contests/AIWIN2020-DEFECT/coco-format'
src_images_dir='/home/ars/sda6/work/contests/mmdetection-aiwin/data/coco/images'
# 用于创建训练集或验证集
phase = 'train'
out_annot_files={
    'train':'train2017.json',
    'val':'val2017.json'
}
out_imgs_dirs={
    'train':'train2017',
    'val':'val2017'
}
# 训练集和验证集划分的界线
train_split=0.8
dataset = {'categories':[],'images':[],'annotations':[]}
# 打开类别标签
with open(os.path.join(root_path, 'classes.txt')) as f:
    classes = f.read().strip().split()

# 建立类别标签和数字id的对应关系
for i, cls in enumerate(classes, 1):
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})



# 读取images文件夹的图片名称
indexes = [f for f in os.listdir(os.path.join(root_path, 'images'))]
random.shuffle(indexes)
num_train=int(len(indexes)*train_split)

# 判断是建立训练集还是验证集
if phase == 'train':
    indexes = [line for i, line in enumerate(indexes) if i <= num_train]
elif phase == 'val':
    indexes = [line for i, line in enumerate(indexes) if i > num_train]

# 读取Bbox信息
with open(os.path.join(root_path, 'annos.txt')) as tr:
    annos = tr.readlines()

for k, index in enumerate(indexes):
    # 用opencv读取图片，得到图像的宽和高
    im = cv2.imread(os.path.join(root_path, 'images/') + index)
    height, width, _ = im.shape
    img=src_images_dir+'/'+index
    img2=root_path+"/"+out_imgs_dirs[phase]+'/'+index
    shutil.copy(img,img2)
    # 添加图像的信息到dataset中
    dataset['images'].append({'file_name': index,
                              'id': k,
                              'width': width,
                              'height': height})

for ii, anno in enumerate(annos):
    parts = anno.strip().split()

    # 如果图像的名称和标记的名称对上，则添加标记
    if parts[0] == index:
        # 类别
        cls_id = parts[1]
        # x_min
        x1 = float(parts[2])
        # y_min
        y1 = float(parts[3])
        # x_max
        x2 = float(parts[4])
        # y_max
        y2 = float(parts[5])
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        dataset['annotations'].append({
            'area': width * height,
            'bbox': [x1, y1, width, height],
            'category_id': int(cls_id),
            'id': i,
            'image_id': k,
            'iscrowd': 0,
            # mask, 矩形是从左上角点按顺时针的四个顶点
            'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
        })

# 保存结果的文件夹
folder = os.path.join(root_path, 'annotations')
if not os.path.exists(folder):
  os.makedirs(folder)
json_name = os.path.join(root_path, 'annotations/%s'%(out_annot_files[phase]))
with open(json_name, 'w') as f:
  json.dump(dataset, f)

