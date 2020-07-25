from wpcv import plp as plp
from wpcv.models.quadnet import QuadNet,heatmap_loss
import os,glob

def quadnet():
	trainer=plp.KeypointDetectorTrainer()
	model=QuadNet()
	trainer.setParams(plp.Params(
		model=model,
		criterion=heatmap_loss,
		train_data_loader=''
	))
