from wpcv import plp
from wpcv.plp import trainers,Params

trainer=trainers.resnet(data_dir='/home/ars/sda5/data/chaoyuan/datasets/classify_datasets/公章')

@trainer.bind_it('train_start')
def hi(trainer):
	trainer.settings.print()
	trainer.state.print1()

y=trainer.train(num_epochs=30,device='cuda')
print(y)