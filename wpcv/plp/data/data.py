import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

def simple_transforms_for_classify_model(input_size=(224,224)):
	train_transform = transforms.Compose([
		transforms.Resize(input_size),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	val_transform = transforms.Compose([
		transforms.Resize(input_size),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
	return dict(
		train=train_transform,
		val=val_transform,
	)
def simple_dataloaders_for_imagefolder(data_dir=None,train_dir=None,val_dir=None,data_transforms=None,train_transform=None,val_transform=None,batch_size=8,shuffle=True):
	train_dir=train_dir or data_dir+'/train'
	val_dir=val_dir or data_dir+'/val'
	train_transform=train_transform or data_transforms['train']
	val_transform=val_transform or data_transforms['val']
	train_dataset = datasets.ImageFolder(root=train_dir,transform=train_transform)
	val_dataset = datasets.ImageFolder(root=val_dir,transform=val_transform)
	train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
	val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
	return dict(
		train=train_data_loader,
		val=val_data_loader,
	)