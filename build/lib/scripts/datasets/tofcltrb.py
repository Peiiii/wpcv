import os,shutil,glob
from scripts.datasets.aiwin import read_bboxes_from_json
def demo():
    labels_dir='/home/ars/sda5/datasets/contests/AIWIN2020-DEFECT/dataset2/Annotations'
    out_file='/home/ars/sda5/datasets/contests/AIWIN2020-DEFECT/coco-format/annos.txt'
    fs=glob.glob(labels_dir+'/*.json')
    text=[]
    classes=['sy','gy','lk']
    label2id={cls:classes.index(cls) for cls in classes}
    for i,f in enumerate(fs):
        name=os.path.basename(f).replace('.json','.jpeg')
        bboxes=read_bboxes_from_json(f)
        for box,label in bboxes:
            txt=' '.join([name,str(label2id[label]),*[str(x) for x in box]])
            text.append(txt)
    text='\n'.join(text)
    with open(out_file,'w') as f:
        f.write(text)




if __name__ == '__main__':
    demo()