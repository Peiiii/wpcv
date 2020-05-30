import json

classes=['sy','gy','lk']
def read_bboxes_from_json(path):
    with open(path,'r') as f:
        dic=json.load(f)
    objects=dic['shapes']
    boxes=[]
    for object in objects:
        points=object['points']
        points.sort(key=lambda p:p[0]+p[1])
        (x1,y1),(x2,y2)=points
        label=object['label']
        box=([x1,y1,x2,y2],label)
        boxes.append(box)
    return boxes