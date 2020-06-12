import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os


def main():
    train_resnet(
        model_type='resnet18',
        input_size=(224, 224),
        batch_size=32,
        num_epochs=50,
        data_dir='/home/ars/sda5/data/chaoyuan/datasets/classify_datasets/公章'
    )


def train_resnet(model_type='resnet18', input_size=(224, 224), batch_size=16,
                 num_epochs=50, data_dir=None, train_dir=None, val_dir=None):
    train_dir = train_dir or data_dir + '/train'
    val_dir = val_dir or data_dir + '/val'
    data_dir = dict(train=train_dir, val=val_dir)
    data_transform = dict(
        train=transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.3, 1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        val=transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    )
    from torchvision.models.resnet import resnet18, resnet50, resnet34, resnet101, resnet152, resnext101_32x8d, \
        resnext50_32x4d
    resnet_models = dict(
        resnet18=resnet18, resnet34=resnet34, resnet50=resnet50,
        resnet101=resnet101, resnet152=resnet152, resnext50_32x4d=resnext50_32x4d, resnext101_32x8d=resnext101_32x8d
    )
    model = resnet_models[model_type](pretrained=True)
    model = ClassifierModel(model)
    data_loaders = {
        phase: DataLoader(dataset=datasets.ImageFolder(root=data_dir[phase], transform=data_transform[phase]),
                          batch_size=batch_size, shuffle=True, num_workers=8)
        for phase in ['train', 'val']
    }
    model.train(data_loaders['train'], data_loaders['val'], num_epochs=num_epochs)


class LoggerFile:
    def __init__(self, path, recreate=False):
        self.path = path
        if recreate:
            self.recreate()

    def recreate(self):
        with open(self.path, 'w') as f:
            pass

    def _push(self, text):
        with open(self.path, 'a') as f:
            f.write(text)

    def log(self, *msgs):
        print(*msgs)
        self.logtofile(*msgs)

    def logtofile(self, *msgs):
        msgs = [str(msg) if hasattr(msg, '__str__') else repr(msg) for msg in msgs]
        msgs = ' '.join(msgs) + '\n'
        self._push(msgs)


class ClassifierModel:
    def __init__(self, model, pretrained=None, strict=True, gpu=True, logfile_path=None,use_tqdm=False):
        self.logfile_path = logfile_path or 'log.txt'
        assert isinstance(model, torch.nn.Module)
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
        print('Device: %s' % (self.device))
        if pretrained:
            self.model.load_state_dict(torch.load(pretrained), strict=strict)
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        try:
            import tqdm
            if use_tqdm:
                self.tqdm = tqdm.tqdm
            else:
                self.tqdm = lambda x: x
        except:
            self.tqdm = lambda x: x

    def train(self, train_dataloader, val_dataloader, num_epochs, lr=0.001, val_interval=1,  save_interval=None,
              save_path=None, val_save_path=None, logfile_path=None):
        if not save_path and not val_save_path and not os.path.exists('weights'):
            os.makedirs('weights')
        save_path = save_path or 'weights/model.pkl'
        val_save_path = val_save_path or 'weights/model_{epoch}_{val_acc:.4f}.pkl'
        logfile_path = logfile_path or self.logfile_path or './train.log'
        logger = LoggerFile(logfile_path)

        device = self.device
        model = self.model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = self.criterion
        best_acc=0

        for epoch in range(num_epochs):
            model.train()
            train_loss = []
            num_total = num_correct = 0
            for inputs, labels in self.tqdm(train_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

                num_total += len(preds)
                num_correct += int(torch.sum(preds == labels))
            train_loss = sum(train_loss) / len(train_loss)
            train_acc = num_correct / (num_total)
            logger.log('Epoch: %s\tLoss: %.4f\tAccuracy: %.4f' % (epoch, train_loss, train_acc))
            if save_interval and epoch % save_interval == 0 and epoch:
                path = save_path.format(epoch=epoch, train_loss=train_loss, train_acc=train_acc)
                torch.save(model.state_dict(), path)
                logger.logtofile('Save model to %s' % (path))
            if val_interval and  epoch % val_interval == 0 and epoch:
                result = self.val(val_dataloader, epoch=epoch, logfile_path=logfile_path)
                val_loss, val_acc = [result[key] for key in ['val_loss', 'val_acc']]
                if val_acc>best_acc:
                    path = val_save_path.format(epoch=epoch, val_loss=val_loss, val_acc=val_acc)
                    torch.save(model.state_dict(), path)
                    logger.log('*'*100)
                    logger.log('Save model to %s ,Val Loss :%.4f\t Val Accuracy: %.4f' % (path,val_loss,val_acc))
                    best_acc=val_acc

    def val(self, dataloader, epoch=None, logfile_path=None):
        logfile_path = logfile_path or self.logfile_path or './val.log'
        logger = LoggerFile(logfile_path)

        device = self.device
        model = self.model
        model.eval()
        criterion = self.criterion
        loss_batch = []
        num_total = num_correct = 0
        for inputs, labels in self.tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            loss_batch.append(loss.item())

            num_total += len(preds)
            num_correct += int(torch.sum(preds == labels))
        val_loss = sum(loss_batch) / len(loss_batch)
        val_acc = num_correct / (num_total)
        # logger.log('Val\tLoss: %.4f\tAccuracy: %.4f' % (val_loss, val_acc))
        return dict(val_acc=val_acc, val_loss=val_loss)


if __name__ == '__main__':
    main()
