from wpcv import plp
from wpcv.plp import trainers

trainer=trainers.resnet(data_dir='/home/ars/sda5/data/chaoyuan/datasets/classify_datasets/公章')
y=trainer.train(num_epochs=1,device='cuda')
print(y)