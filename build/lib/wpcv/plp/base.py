import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import os, inspect, functools, logging
from logging import basicConfig
from wk import PointDict
from wk.debug.logger import LoggerFile


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
		elif self.lr_scheduler is None:
			self.lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
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


class Event(PointDict):
	def __init__(self, name='event', argspace={}):
		# super().__init__(name)
		self.name = name
		# self.update(**attrs)

		assert isinstance(argspace, (ArgumentSpace, dict))
		if not isinstance(argspace, ArgumentSpace):
			argspace = ArgumentSpace(**argspace)
		self.argspace = argspace


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
				assert isinstance(event, Event)
				# data_source = {}
				# data_source.update(**event.attrs)
				# data_source.update(event=event)
				data_source = event.argspace
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
		if isinstance(event, str): event = Event(event)
		res = None
		for e, funcs in self.listener_dict.items():
			if e == event.name:
				for func in funcs:
					res = func(event)
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


class ArgumentSpace(PointDict):
	class EmptyArgument:
		pass
	class EmptyKey:
		pass

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.seta(__parent__={})

	@classmethod
	def make(cls, *args, **kwargs):
		for arg in args:
			assert isinstance(arg, dict)
		args = list(args)
		args.append(kwargs)

		def my_update(dic1, *dics):
			for dic in dics:
				for k, v2 in dic.items():
					v1 = dic1.get(k, None)
					v = v2 if v2 is not None else v1
					dic1[k] = v
			return dic1

		new = my_update(*args)
		return cls(**new)

	def get_parent(self):
		return self.geta('__parent__')

	def detach_parent(self):
		parent = self.geta('__parent__')
		self.seta(__parent__={})
		return parent

	def set_parent(self, parent):
		assert isinstance(parent, ArgumentSpace)
		self.seta(__parent__=parent)

	def get_argument(self, arg, default=EmptyArgument):
		return self.get(arg, default)

	def get(self, arg, default=EmptyArgument):
		EmptyKey=ArgumentSpace.EmptyKey
		v = PointDict.get(self, arg, EmptyKey)
		if v is EmptyKey or v is None:
			if isinstance(self.geta('__parent__'), ArgumentSpace):
				v = self.geta('__parent__').get_argument(arg, EmptyKey)
				# print('parent v:',v)
				if v is EmptyKey:
					# print('v is Empty')
					if default is ArgumentSpace.EmptyArgument:
						raise Exception('Cannot get argument %s' % (arg))
					v=default
				# else:
				# 	print('V is not Empty')
		# print(arg,v,default)
		return v

	def retrieve_arguments(self, args, strict=True):
		Empty=EmptyKey=ArgumentSpace.EmptyKey

		arg_list = []
		params = []

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
				v = self.get_argument(args, Empty)
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
			params.append(handle_if_empty(self.get_argument(arg, Empty), arg))
		return params


class Callback:
	pass


class DefaultCallback(Callback):
	# @staticmethod
	def on_train_start(self, device):
		print("Start training, device:%s" % (device))

	def on_epoch_train_start(self, trainState):
		pass

	def on_batch_train_end(self, state, trainState):
		pass

	def on_epoch_train_end(self, state, trainState):
		pass

	def on_train_end(self, ):
		print("Training process finished.")

	def on_val_start(self, state, valState):
		pass

	def on_batch_val_end(self, state, valState):
		pass

	def on_val_end(self, state, valState):
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
		self.argument_space = ArgumentSpace()
	def load_state_dict(self,state_dict):
		if isinstance(state_dict,str):
			state_dict=torch.load(state_dict,map_location=self.params.device)
		self.params.model.load_state_dict(state_dict)
	def setup(self):
		'''
		设置参数完成后才能setup,
		setup后才能运行函数,如train,val,test,predict
		:return:
		'''
		self.params.setup()
		self.settings.setup()
		self.bind_callback(self)
		self.event_manager.bind('epoch_train_end', self.val)
		self.event_manager.bind('trigger_val', self.val)
		if self.use_default_logger:
			self.bind_callback(DefaultCallback())
		self.argument_space = ArgumentSpace.make(self.argument_dict())
		return self

	def log(self, *args, **kwargs):
		print(*args, **kwargs)


	def emit(self, name, argspace=None, **kwargs):
		assert argspace is None or isinstance(argspace, ArgumentSpace)
		dics = [argspace] if argspace is not None else []
		dics.append(kwargs)
		argspace = ArgumentSpace.make(*dics)
		'''
		emit 的参数会在emit期间有用,在emit后失效
		'''
		argspace.set_parent(self.argument_space)
		self.argument_space = argspace
		res = self.event_manager.emit(Event(name, self.argument_space))
		self.argument_space = argspace.detach_parent()
		# print("detached:",self.argument_space)
		return res

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
	def bind(self,event,callback):
		self.event_manager.bind(event,callback)
		return self
	def bind_it(self,event):
		return self.event_manager.bind_this(event)

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

	def calculate_loss(self, preds, labels, criterion, device):
		labels = labels.to(device)
		loss = criterion(preds, labels)
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
		self.log('Epoch:{epoch}  Loss:{loss:.4f}  Learningrate:{lr:.4f}'.format(epoch=state.epoch,
		                                                                        loss=state.trainState.epoch_loss,
		                                                                        lr=state.trainState.lr))

	def overall_summary(self):
		pass

	def val(self, **kwargs):
		return self.emit('val', **kwargs)

	def on_val(self, model, device, val_data_loader, criterion,with_tqdm):
		state, valState = self.retrieve_arguments('state,valState')
		state.phase = 'val'
		valState.clear_state()
		model.to(device)
		model.eval()
		self.emit('val_start')
		if with_tqdm:
			import tqdm
			val_data_loader=tqdm.tqdm(val_data_loader)
		for valState.step, (inputs, labels) in enumerate(val_data_loader):
			self.emit('batch_val_start')
			valState.inputs, valState.labels = self.to_device(inputs, labels, device)
			valState.preds = self.forward_step(valState.inputs)
			loss = self.calculate_loss(valState.preds, valState.labels, criterion, device)
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
		model.train()
		state.phase = 'train'

	def argument_dict(self):
		'''
		把常用的参数都封装进一个字典里面，方便获取参数
		不能把argument_space放进去，因为它是动态变化的，不同时间可能时不同的对象。因为在emit的时候会对它进行修改和替换
		:return:
		'''
		params, settings, state, flags = self.params, self.settings, self.state, self.flags
		# trainState = state.trainState
		# valState=state.valState
		# model, device, train_data_loader, val_data_loader, criterion, optimizer, lr_scheduler = params.model, params.device, params.train_data_loader, params.val_data_loader, params.criterion, params.optimizer, params.lr_scheduler
		dic = {}
		for bank in [params, settings, state]:
			dic.update(**bank)
		dic.update(
			# params=params,
			# settings=settings,
			trainer=self,
			state=state,
			flags=flags,
		)
		return dic

	def retrieve_arguments(self, args, strict=True):
		'''example:
		a,b,c=self.retrieve_arguments('a,b,c')
		'''
		return self.argument_space.retrieve_arguments(args, strict=strict)

	# def train(self,**kwargs):
	def auto_feed_params(self,func):
		arg_dict=get_arg_dict(func)
		@functools.wraps(func)
		def wrapper(*args,**kwargs):
			params = {}
			for k, v_default in arg_dict:
				v = self.argument_space.get(k, None)
				if v is None:
					v = v_default if not v_default is not inspect._empty else None
				params[k] = v
			res = func(*args,**params,**kwargs)
			return res
		return wrapper
	def test(self):
		raise NotImplementedError
	def predict(self,x):
		raise NotImplementedError
	def train(self, model=None, device=None, train_data_loader=None, val_data_loader=None, criterion=None,
	          optimizer=None, lr_scheduler=None, start_global_step=None, start_epoch=None, num_epochs=None,**kwargs):
		'''
		You have three ways to call train:
		call trainer.train with or without arguments
		call trainer.emit('train')
		call trainer.on_train with arguments
		'''
		kwargs = dict(
			model=model, device=device, train_data_loader=train_data_loader, val_data_loader=val_data_loader,
			optimizer=optimizer, lr_scheduler=lr_scheduler, start_global_step=start_global_step,
			start_epoch=start_epoch, num_epochs=num_epochs,**kwargs
		)
		return self.emit('train', **kwargs)
	def lr_scheduler_step(self):
		self.emit('lr_scheduler_step')
	def on_lr_scheduler_step(self,lr_scheduler,epoch,trainState):
		lr_scheduler.step(trainState.epoch_loss,epoch)
	def on_train(self, model, device, train_data_loader, val_data_loader, criterion, optimizer, lr_scheduler,
	             start_global_step, start_epoch, num_epochs,with_tqdm):
		'''
		Because only state and flags need to be changed during the process, we should only get state and flags
		directly from Trainer object, for other arguments like params and settings, they don't need to be change so we
		should make it ReadOnly. And because  they have no states that need to be stored, so there should be a way to
		specify these arguments by users when the functions are called.
		'''
		state, flags, trainState = self.retrieve_arguments('state, flags,trainState')
		model.to(device)
		state.phase = 'train'
		state.global_step = start_global_step
		self.emit('train_start')
		if with_tqdm:
			import tqdm
		for state.epoch in range(start_epoch, start_epoch + num_epochs):
			trainState.clear_state()
			model.train()
			self.emit('epoch_train_start')
			if with_tqdm:
				train_data_loader=tqdm.tqdm(train_data_loader)
			for trainState.step, (inputs, labels) in enumerate(train_data_loader):
				self.emit('batch_train_start')
				trainState.lr = optimizer.state_dict()['param_groups'][0]['lr']
				trainState.inputs, trainState.labels = self.to_device(inputs, labels, device)
				trainState.preds = self.forward_step(trainState.inputs)
				optimizer.zero_grad()
				loss = self.calculate_loss(trainState.preds, trainState.labels, criterion, device)
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
				self.lr_scheduler_step()
			if flags.stop_trainning:
				break
		self.emit('train_end')
		self.overall_summary()
		return dict(
			message='Dear passenger, the journey of training has ended.'
		)


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

class KeypointDetectorTrainer(Trainer):
	pass

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
