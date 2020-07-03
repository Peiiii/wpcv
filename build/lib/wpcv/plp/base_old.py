import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import os, inspect, functools, logging
from logging import basicConfig
from wk import PointDict
from wk.debug.logger import LoggerFile


class Event(PointDict):
	def __init__(self, name='event', attrs={}):
		# super().__init__(name)
		self.name = name
		self.update(**attrs)


class Params(PointDict):
	def __init__(self, model=None, device=None, train_data_loader=None, val_data_loader=None, criterion=None,
	             optimizer=None, lr_scheduler=None):
		self.model = model
		self.device = device
		self.criterion = criterion
		self.optimizer = optimizer
		self.lr_scheduler = lr_scheduler
		self.train_data_loader = train_data_loader
		self.val_data_loader = val_data_loader
		self.dataloaders = None

	def setup(self):
		assert self.model
		assert isinstance(self.model, torch.nn.Module)
		if isinstance(self.device, str):
			self.device = torch.device(self.device)
		elif isinstance(self.device, torch.device):
			pass
		else:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		if self.criterion is None:
			self.criterion = torch.nn.CrossEntropyLoss()
		assert self.criterion
		if isinstance(self.optimizer, dict):
			optim_type = self.optimizer.pop('type')
			if isinstance(optim_type, str):
				optims = {'Adam': torch.optim.Adam, 'SGD': torch.optim.SGD, }
				optim_type = optims[optim_type]
			self.optimizer = optim_type(self.model.parameters(), **self.optimizer)
		elif self.optimizer is None:
			self.optimizer = torch.optim.Adam(self.model.parameters())
		else:
			assert self.optimizer
		if isinstance(self.lr_scheduler, dict):
			lr_sch_type = self.lr_scheduler.pop('type')
			if isinstance(lr_sch_type, str):
				lr_sch_types = {'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
				                'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR, }
				lr_sch_type = lr_sch_types[lr_sch_type]
			self.lr_scheduler = lr_sch_type(self.optimizer, **self.lr_scheduler)
		assert self.criterion
		self.train_data_loader = self.train_data_loader or self.dataloaders['train']
		if not self.val_data_loader and self.dataloaders:
			self.val_data_loader = self.dataloaders['val']
		assert self.train_data_loader


class Settings(PointDict):
	def __init__(self, num_epochs=200, start_epoch=0, start_gloabl_step=0, monitor='val_acc', save_best=True,
	             mode='train_val',
	             model_best_path='weights/model_best.pth',
	             auto_make_dirs=True, save_interval=None, val_interval=1, model_save_path='weights/model.pth',
	             save_val_model=False, val_model_save_path='weights/model_{epoch}_{val_acc:.3%}.pth'):
		super().__init__()
		self.num_epochs = num_epochs
		self.start_global_step = start_gloabl_step
		self.start_epoch = start_epoch
		self.monitor = monitor
		self.mode = mode
		self.save_best = save_best
		self.model_best_path = model_best_path
		self.auto_make_dirs = auto_make_dirs

	def setup(self):
		pass


def get_arg_dict(func):
	sign = inspect.signature(func)
	keys = list(sign.parameters.keys())
	dic = dict()
	# print(func.__name__,keys,'is_bound:%s'%(is_bound(func)))
	# if is_bound(func) and not isinstance(func, staticmethod):
	# 	keys = keys[1:]
	for key in keys:
		value = sign.parameters.get(key).default
		dic[key] = value
	# print(dic)
	return dic


def is_bound(m):
	return hasattr(m, '__self__')


def get_callback_listeners(obj):
	# print(obj)
	# dic = inspect.getmembers(obj, predicate=inspect.isfunction)
	dic = inspect.getmembers(obj, predicate=inspect.ismethod)
	funcs = []
	for k, v in dic:
		if k.startswith('on_'):
			funcs.append(v)
	return funcs


class EventManger:
	def __init__(self):
		self.listener_dict = {}
		self.source_map = {}

	def bind(self, event, callbacks: 'iterable or callable'):
		import functools, inspect
		if not isinstance(callbacks, (list, tuple, set)):
			callbacks = [callbacks]
		if event not in self.listener_dict.keys():
			self.listener_dict[event] = []
			self.source_map[event] = []
		for callback in callbacks:
			func = callback
			if func in self.source_map[event]:
				logging.warning("Function %s has already been bound, cannot be bound again." % (func.__name__))
				continue
			else:
				logging.warning("Function %s has already been bound." % (func.__name__))
				self.source_map[event].append(func)
			# argspec = inspect.getfullargspec(func)
			# args = argspec.args
			arg_dict = get_arg_dict(func)
			args = list(arg_dict.keys())

			# print(args)
			@functools.wraps(func)
			def wrapper(event):
				data_source = {}
				data_source.update(**event)
				data_source.update(event=event)
				params = {}
				for arg in args:
					v = data_source.get(arg, None)
					if v is None:
						v = arg_dict.get(v)
						if v is inspect._empty:
							v = None
					params[arg] = v
				# print(params)
				res = func(**params)
				return res

			self.listener_dict[event].append(wrapper)

	def bind_this(self, event):
		def decorator(func):
			self.bind(event, func)
			return func

		return decorator

	def emit(self, event: str or Event):
		assert isinstance(event, (Event, str))
		res=None
		for e, funcs in self.listener_dict.items():
			if e == event.name:
				for func in funcs:
					res=func(event)
		return res


class META:
	class Extra(PointDict):
		def __init__(self):
			super().__init__()

	class Pipe(PointDict):
		def __init__(self):
			super().__init__()

		def push(self, dic):
			self.update(**dic)

	class ClassifyMetrix(PointDict):
		def __init__(self, trainer):
			super().__init__()
			self.trainer = trainer
			self.sample_count = None
			self.pred_count = None
			self.correct_count = None
			self.acc_history = []
			self.best_accuracy = 1e-7
			self.stucks = 0
			self.accuracy = None
			self.recalls = None
			self.precisions = None

		def push(self, dic):
			for k, v in dic.items():
				if self.get(k) is None:
					self[k] = v
				else:
					self[k] += v

		def reset_epoch(self):
			self.sample_count = None
			self.pred_count = None
			self.correct_count = None

		def analyze(self):
			# print('Analyzing..., phase:%s'%(self.trainer.state.phase))
			def non_zero(t):
				# print(t)
				# assert isinstance(t,torch.Tensor)
				mask = t == 0
				epsilon = 1e-7
				t = t + torch.zeros_like(mask).fill_(epsilon) * mask
				# print(t)
				return t

			recalls = self.correct_count / non_zero(self.sample_count)
			precisions = self.correct_count / non_zero(self.pred_count)
			num_corrects = self.correct_count.sum()
			num_samples = self.sample_count.sum()
			# print('corrects:%s,samples:%s'%(num_corrects,num_samples))
			recall = self.correct_count.sum() / non_zero(self.sample_count.sum())
			accuracy = recall.item()

			self.acc_history.append(accuracy)
			if not self.best_accuracy:
				self.best_accuracy = accuracy
			else:
				if accuracy > self.best_accuracy:
					old = self.best_accuracy
					self.best_accuracy = accuracy
					self.stucks = 0
					self.trainer.emit('best_accuracy', best_accuracy=self.best_accuracy, old_best_accuracy=old)
				else:
					self.stucks += 1
			result = dict(
				recalls=recalls.numpy().tolist(), precisions=precisions.numpy().tolist(), accuracy=accuracy
			)
			self.update(**result)
			return result

	class RunningState(PointDict):
		def __init__(self, trainer):
			super().__init__()
			self.trainer = trainer
			self.step = 0
			self.batch_losses = []
			self.batch_loss = None
			self.inputs = None
			self.labels = None
			self.preds = None
			self.extra = META.Extra()
			self.metrix = META.ClassifyMetrix(trainer=self.trainer)

		def clear_state(self):
			'''reset after each epoch'''
			self.step = 0
			self.batch_losses = []
			self.batch_loss = None
			self.inputs = None
			self.labels = None
			self.preds = None
			self.metrix.reset_epoch()

		def end_batch(self):
			self.batch_losses.append(self.batch_loss)

		def end_epoch(self):
			self.epoch_loss = np.mean(self.batch_losses)

	class ValState(RunningState):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)

			self.batch_accs = []
			self.batch_acc = None
			self.epoch_loss = None
			self.epoch_acc = None

		def clear_state(self):
			META.RunningState.clear_state(self)
			self.batch_accs = []
			self.batch_acc = None
			self.epoch_loss = None
			self.epoch_acc = None

	class TrainState(RunningState):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.batch_accs = []
			self.batch_acc = None
			self.epoch_loss = None
			self.epoch_acc = None
			self.lr = None

		def clear_state(self):
			META.RunningState.clear_state(self)
			self.batch_accs = []
			self.batch_acc = None
			self.epoch_loss = None
			self.epoch_acc = None


# self.val = META.ValState()


class State(PointDict):
	def __init__(self, trainer):
		self.phase = None
		self.epoch = None
		self.global_step = None
		self.trainState = META.TrainState(trainer)
		self.valState = META.ValState(trainer)
		self.extra = META.Extra()
		self.pipe = META.Extra()
		self.trainer = trainer


class Flags(PointDict):
	def __init__(self):
		super().__init__()
		self.stop_trainning = False


class Callback:
	pass


class DefaultCallback(Callback):
	# @staticmethod
	def on_train_start(self, params):
		print("Start training, device:%s" % (params.device))

	def on_epoch_train_start(self, trainState):
		pass

	def on_batch_train_end(self, state, trainState):
		# trainState.batch_losses.append(trainState.batch_loss)
		pass

	def on_epoch_train_end(self, state, trainState):
		# trainState.epoch_loss = np.min(trainState.batch_losses)
		# trainer.emit('trigger_val')
		pass

	def on_train_end(self, ):
		print("Training process finished.")

	def on_val_start(self, state, valState):
		# print('Start valing...')
		pass

	def on_batch_val_end(self, state, valState):
		# valState.batch_losses.append(valState.batch_loss)
		pass

	def on_val_end(self, state, valState):
		# valState.epoch_loss= np.mean(valState.batch_losses)
		# print('ValLoss:%.4f' % (valState.epoch_loss))
		pass

	def on_trigger_val(self, state, trainer):
		assert isinstance(trainer, Trainer)


class Trainer:
	def __init__(self, use_default_logger=True):
		self.params = Params()
		self.settings = Settings()
		self.state = State(trainer=self)
		self.flags = Flags()
		self.event_manager = EventManger()
		self.use_default_logger = use_default_logger

	def setup(self):
		self.params.setup()
		self.settings.setup()
		self.bind_callback(self)
		self.event_manager.bind('epoch_train_end', self.val)
		self.event_manager.bind('trigger_val', self.val)
		if self.use_default_logger:
			self.bind_callback(DefaultCallback())
		return self

	def log(self, *args, **kwargs):
		print(*args, **kwargs)

	def load_state_dict(self, state_dict):
		if isinstance(state_dict, str):
			state_dict = torch.load(state_dict)
		self.params.model.load_state_dict(state_dict)

	def emit(self, name, **kwargs):
		environment = self.argument_dict()
		environment.update(**kwargs)
		return self.event_manager.emit(Event(name, attrs=environment))

	def bind_callback(self, callbacks):
		if not isinstance(callbacks, (list, tuple, set)):
			callbacks = [callbacks]
		for callback in callbacks:
			funcs = get_callback_listeners(callback)
			self.quick_bind(funcs)
		return self

	def quick_bind(self, funcs):
		if not isinstance(funcs, (list, tuple, set)):
			funcs = [funcs]
		for func in funcs:
			if func.__name__.startswith('on_'):
				event = func.__name__.split('_', maxsplit=1)[-1]
				self.event_manager.bind(event, func)
			else:
				raise Exception('Name does not start with "on" : %s' % (func.__name__))
		return self

	def auto_bind(self):
		logging.info('Finding potential listeners(functions with names start with "on_")...')
		dic = globals()
		for k, v in dic.items():
			if inspect.isfunction(v):
				func = v
				if func.__name__.startswith('on_'):
					event = func.__name__.split('_', maxsplit=1)[-1]
					self.event_manager.bind(event, func)
		return self

	def setParams(self, params):
		self.params.update(**params)
		return self

	def setSettings(self, settings):
		self.settings.update(**settings)
		return self

	def getState(self):
		return self.state

	def to_device(self, inputs, labels, device):
		inputs = inputs.to(device)
		labels = labels.to(device)
		return inputs, labels

	def forward_step(self, inputs):
		preds = self.params.model(inputs)
		return preds

	def calculate_loss(self, preds, labels):
		labels = labels.to(self.params.device)
		loss = self.params.criterion(preds, labels)
		return loss

	def eval_batch(self, runState):
		assert isinstance(runState, META.RunningState)

	def eval_epoch(self, runState):
		assert isinstance(runState, META.RunningState)

	def batch_summary(self):
		pass

	def epoch_val_summary(self, state):
		log = 'ValLoss:{val_loss:.4f}'.format(val_loss=state.valState.epoch_loss)
		self.log(log)

	def epoch_train_summary(self, state):
		pass

	def epoch_summary(self, state):
		print('Epoch:{epoch}  Loss:{loss:.4f}  Learningrate:{lr:.4f}'.format(epoch=state.epoch,
		                                                                     loss=state.trainState.epoch_loss,
		                                                                     lr=state.trainState.lr))

	def overall_summary(self):
		pass

	def val(self, valState=None, params=None, state=None):
		state = state or self.state
		params = params or self.params
		valState = valState or state.valState
		state.phase = 'val'
		valState.clear_state()
		params.model.to(params.device)
		params.model.eval()
		self.emit('val_start')
		for valState.step, (inputs, labels) in enumerate(params.val_data_loader):
			self.emit('batch_val_start')
			valState.inputs, valState.labels = self.to_device(inputs, labels, params.device)
			valState.preds = self.forward_step(valState.inputs)
			loss = self.calculate_loss(valState.preds, valState.labels)
			valState.batch_loss = loss.item()
			valState.end_batch()
			self.eval_batch(valState)
			self.emit('batch_val_end')
			self.batch_summary()
		valState.end_epoch()
		result = self.eval_epoch(valState) or {}
		self.emit('val_eval_finished', **result)
		self.epoch_val_summary(state)
		self.emit('val_end')
		params.model.train()
		state.phase = 'train'

	def argument_dict(self):
		params, settings, state, flags = self.params, self.settings, self.state, self.flags
		# trainState = state.trainState
		# valState=state.valState
		# model, device, train_data_loader, val_data_loader, criterion, optimizer, lr_scheduler = params.model, params.device, params.train_data_loader, params.val_data_loader, params.criterion, params.optimizer, params.lr_scheduler
		dic = {}
		for bank in [params, settings, state]:
			dic.update(**bank)
		dic.update(
			params=params,
			settings=settings,
			state=state,
			flags=flags,
		)
		return dic

	def retrieve_arguments(self, args, strict=True):
		class Empty:
			pass

		arg_list = []
		params = []
		arguments = self.argument_dict()

		def handle_if_empty(arg, name=''):
			if arg is not Empty:
				return arg
			if strict:
				raise Exception('Cannot retrieve argument %s.' % (name))
			return None

		def check_arg_name(txt):
			return True

		if isinstance(args, str):
			if not ',' in args:
				v = arguments.get(args, Empty)
				return handle_if_empty(v, args)
			else:
				args = args.strip().strip(',').strip().split(',')
				for arg in args:
					arg = arg.strip()
					check_arg_name(arg)
					arg_list.append(arg)
		else:
			assert isinstance(args, (list, tuple, set))
			arg_list = args
		for arg in arg_list:
			params.append(handle_if_empty(arguments.get(arg, Empty), arg))
		return params

	def train(self):
		'''model,criterion,optimizer,lr_scheduler,data_loader
		epoch_loss,epoch_acc,epoch_loss,epoch_acc,
		val_model,save_model
		'''
		# params, settings, state, flags = self.params, self.settings, self.state, self.flags
		params, settings, state, flags = self.retrieve_arguments('params, settings, state, flags')
		# trainState = state.trainState
		trainState = self.retrieve_arguments('trainState')
		# model, device, train_data_loader, val_data_loader, criterion, optimizer, lr_scheduler = params.model, params.device, params.train_data_loader, params.val_data_loader, params.criterion, params.optimizer, params.lr_scheduler
		model, device, train_data_loader, val_data_loader, criterion, optimizer, lr_scheduler = self.retrieve_arguments(
			'model, device, train_data_loader, val_data_loader, criterion, optimizer, lr_scheduler')
		model.to(device)
		state.phase = 'train'
		state.global_step = settings.start_global_step
		self.emit('train_start')
		for state.epoch in range(settings.start_epoch, settings.start_epoch + settings.num_epochs):
			trainState.clear_state()
			model.train()
			self.emit('epoch_train_start')
			for trainState.step, (inputs, labels) in enumerate(train_data_loader):
				self.emit('batch_train_start')
				trainState.lr = optimizer.state_dict()['param_groups'][0]['lr']
				trainState.inputs, trainState.labels = self.to_device(inputs, labels, device)
				trainState.preds = self.forward_step(trainState.inputs)
				optimizer.zero_grad()
				loss = self.calculate_loss(trainState.preds, trainState.labels)
				loss.backward()
				optimizer.step()
				trainState.batch_loss = loss.item()
				trainState.end_batch()  # arrange info
				self.eval_batch(trainState)  # do some eval

				self.emit('batch_train_end')
				self.batch_summary()
				state.global_step += 1
			trainState.end_epoch()
			self.eval_epoch(trainState)  # do some eval
			self.emit('epoch_train_end')
			self.epoch_train_summary(state)
			self.epoch_summary(state)
			if lr_scheduler:
				lr_scheduler.step()
			if flags.stop_trainning:
				break
		self.emit('train_end')
		self.overall_summary()


class SaveCallback(Callback):
	def on_best_accuracy(self, state, model, epoch, best_accuracy, old_best_accuracy, mode):
		if mode == 'val':
			return
		if state.phase == 'val':
			print('New best accuracy: %.4f improved from %.4f , model saved.' % (
				best_accuracy, old_best_accuracy if old_best_accuracy is not None else 0))
			torch.save(model.state_dict(), 'model_best.pkl'.format(epoch=epoch))

	def on_val_end(self, model, epoch, mode):
		if mode == 'val':
			return
		torch.save(model.state_dict(), 'model.pkl'.format(epoch=epoch))


class EarlyStopping(Callback):
	def __init__(self, patience=10):
		self.acc_history = []
		self.best_history = 0
		self.stucks = 0
		self.patience = patience

	def on_val_eval_finished(self, accuracy, flags):
		if accuracy > self.best_history:
			self.best_history = accuracy
			self.stucks = 0
		else:
			self.stucks += 1
		if self.stucks > self.patience:
			flags.stop_trainning = True
			print("Not improve for %s epochs, going to stop trainning." % (self.stucks))


class ClassifierTrainer(Trainer):
	def eval_batch(self, runState):
		preds, labels = runState.preds.cpu(), runState.labels.cpu()
		batch_size, num_classes = preds.shape
		_, preds = torch.max(preds, 1)
		labels = torch.zeros((batch_size, num_classes)).scatter_(-1, torch.unsqueeze(labels, -1), 1)
		preds = torch.zeros((batch_size, num_classes)).scatter_(-1, torch.unsqueeze(preds, -1), 1)
		sample_count = torch.sum(labels, 0)
		pred_count = torch.sum(preds, 0)
		correct_count = torch.sum(labels * preds, 0)
		runState.metrix.push(dict(
			sample_count=sample_count,
			pred_count=pred_count,
			correct_count=correct_count,
		))

	def eval_epoch(self, runState):
		result = runState.metrix.analyze()
		self.emit('eval_epoch_finished', **result)
		return result

	def epoch_val_summary(self, state):
		if self.settings.mode == 'val':
			return
		valState = state.valState
		log = 'ValLoss:{val_loss:.4f}  ValAccuracy:{val_acc:.4f}  ValRecalls:{val_recalls}  ValPrecisions:{val_precisions}'.format(
			val_loss=valState.epoch_loss, val_acc=valState.metrix.accuracy, val_recalls=valState.metrix.recalls,
			val_precisions=valState.metrix.precisions)
		self.log(log)

	def epoch_summary(self, state):
		trainState = state.trainState
		valState = state.valState
		log = '''Epoch:{epoch}  Learningrate:{lr:.6f}  Loss:{loss:.4f}  Accuracy:{acc:.4f}  ValLoss:{val_loss:.4f}  ValAccuracy:{val_acc:.4f}'''.format(
			epoch=state.epoch, lr=trainState.lr, loss=trainState.epoch_loss, acc=trainState.metrix.accuracy,
			val_loss=valState.epoch_loss, val_acc=valState.metrix.accuracy,
		)
		self.log(log)


def demo():
	trainer = ClassifierTrainer()
	num_classes = 4
	model = models.resnet18(pretrained=True)
	model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
	train_transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
	])
	val_transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor()
	])
	train_dataset = datasets.ImageFolder(root='/home/ars/sda5/data/chaoyuan/datasets/classify_datasets/公章/train',
	                                     # train_dataset = datasets.ImageFolder(root='/home/ars/sda5/data/chaoyuan/datasets/classify_datasets/文档方向分类/train',
	                                     transform=train_transform)
	val_dataset = datasets.ImageFolder(root='/home/ars/sda5/data/chaoyuan/datasets/classify_datasets/公章/val',
	                                   # val_dataset = datasets.ImageFolder(root='/home/ars/sda5/data/chaoyuan/datasets/classify_datasets/文档方向分类/val',
	                                   transform=val_transform)
	train_data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
	val_data_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
	trainer.setParams(dict(
		model=model,
		train_data_loader=train_data_loader,
		val_data_loader=val_data_loader,
	)).setSettings(dict(
		num_epoch=200,
		mode='val',
	)).setup().bind_callback([
		SaveCallback(), EarlyStopping(patience=20),
	])
	trainer.train()


# trainer.load_state_dict('model_best.pkl')
# trainer.val()


if __name__ == '__main__':
	demo()
