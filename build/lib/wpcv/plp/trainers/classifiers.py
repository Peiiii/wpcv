
from wpcv import plp as plp
import os,glob



def resnet(data_dir=None,name='resnet18',pretrained=True,num_classes='auto',input_size=(224,224),batch_size=8,num_epoch=200,patience=20,shuffle=True):
	trainer = plp.ClassifierTrainer()
	if num_classes=='auto':
		num_classes=len(os.listdir(data_dir+'/train'))
	model=plp.resnet_model(name,pretrained=pretrained,num_classes=num_classes)
	data_transforms=plp.simple_transforms_for_classify_model(input_size=input_size)
	dataloaders=plp.simple_dataloaders_for_imagefolder(
		data_dir=data_dir,
		data_transforms=data_transforms,
		batch_size=batch_size,
		shuffle=shuffle,
	)
	trainer.setParams(dict(
		model=model,
		dataloaders=dataloaders,
	)).setSettings(dict(
		num_epoch=num_epoch,
	)).bind_callback([
		plp.SaveCallback(),plp.EarlyStopping(patience=patience),
	]).setup()
	return trainer

def classifier(data_dir,name='resnet18',pretrained=True,num_classes='auto',input_size=(224,224),batch_size=8,num_epoch=200,patience=20,shuffle=True):
	pass