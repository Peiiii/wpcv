import random


class Transform:
	pass


def random_float_generator(rg):
	if not isinstance(rg, (tuple, list)):
		rg = [-rg, rg]
	low, high = rg

	def func():
		return random.random() * (high - low) + low

	return func


class Identical(object):
	def __call__(self, *args):
		if len(args) == 1:
			return args[0]
		return args


class Compose(object):
	"""Composes several transforms together.

	Args:
		transforms (list of ``Transform`` objects): list of transforms to compose.
	Example:
		>>> transforms.Compose([
		>>>     transforms.CenterCrop(10),
		>>>     transforms.ToTensor(),
		>>> ])
	"""

	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, *args):
		x = args
		for t in self.transforms:
			x = t(*x)
			if not isinstance(x, tuple):
				x = (x,)
		if len(x) == 1:
			x = x[0]
		return x

	def __repr__(self):
		format_string = self.__class__.__name__ + '('
		for t in self.transforms:
			format_string += '\n'
			format_string += '    {0}'.format(t)
		format_string += '\n)'
		return format_string


class Zip(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, *args):
		assert len(args) == len(self.transforms)
		res = []
		for arg, transform in zip(args, self.transforms):
			res.append(transform(arg))
		return tuple(res)


class RandomMultiChoice(object):
	'''
	for each choice , choose it with a given probability,
	which means, multiple of them can be chosen simultaneously
	'''

	def __init__(self, transforms, probs=None):
		self.transforms = transforms
		if not probs:
			probs = [0.5 for t in transforms]
		self.probs = probs

	def __call__(self, *args):
		x = args
		for i in range(len(self.transforms)):
			if random.random() < self.probs[i]:
				x = self.transforms[i](*x)
				if not isinstance(x, tuple):
					x = (x,)
		if isinstance(x, (tuple,)) and len(x) == 1:
			return x[0]
		else:
			return x


class RandomTransforms(object):
	"""Base class for a list of transformations with randomness

	Args:
		transforms (list or tuple): list of transformations
	"""

	def __init__(self, transforms):
		assert isinstance(transforms, (list, tuple))
		self.transforms = transforms

	def __call__(self, *args, **kwargs):
		raise NotImplementedError()

	def __repr__(self):
		format_string = self.__class__.__name__ + '('
		for t in self.transforms:
			format_string += '\n'
			format_string += '    {0}'.format(t)
		format_string += '\n)'
		return format_string


class RandomApply(object):
	"""Apply randomly a list of transformations with a given probability

	Args:
		transforms (list or tuple): list of transformations
		p (float): probability
	"""

	def __init__(self, transform, p=0.5, *args, **kwargs):
		self.transform = transform
		self.p = p
		self.args = args
		self.kwargs = kwargs

	def __call__(self, *args):
		if self.p < random.random():
			return args[0] if len(args) == 1 else args
		x = self.transform(*args, *self.args, **self.kwargs)
		return x


class RandomOrder(RandomTransforms):
	"""Apply a list of transformations in a random order
	"""

	def __call__(self, *args, **kwargs):
		x = args
		order = list(range(len(self.transforms)))
		random.shuffle(order)
		for i in order:
			x = self.transforms[i](*x, **kwargs)
			if not isinstance(x, tuple):
				x = (x,)
		if isinstance(x, (tuple,)) and len(x) == 1:
			return x[0]
		else:
			return x


class RandomChoice(RandomTransforms):
	"""Apply single transformation randomly picked from a list
	"""

	def __call__(self, *args, **kwargs):
		t = random.choice(self.transforms)
		return t(*args, **kwargs)


class RandomChoices(object):
	"""Apply single transformation randomly picked from a list
	"""

	def __init__(self, transforms, k=1, weights=None):
		self.transforms = transforms
		self.k = k
		self.weights = None

	def __call__(self, *args, **kwargs):
		x = args
		ts = random.choices(self.transforms, weights=self.weights, k=self.k)
		for i in range(len(ts)):
			x = self.transforms[i](*x, **kwargs)
			if not isinstance(x, tuple):
				x = (x,)
		if isinstance(x, (tuple,)) and len(x) == 1:
			return x[0]
		else:
			return x
