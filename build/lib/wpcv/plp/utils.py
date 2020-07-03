import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18, resnet50, resnet34, resnet101, resnet152, resnext101_32x8d, \
        resnext50_32x4d
resnet_models = dict(
    resnet18=resnet18, resnet34=resnet34, resnet50=resnet50,
    resnet101=resnet101, resnet152=resnet152, resnext50_32x4d=resnext50_32x4d, resnext101_32x8d=resnext101_32x8d
)

def resnet_model(name='resnet18',pretrained=False,num_classes=None):
	model = resnet_models[name](pretrained=pretrained)
	model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
	return model
