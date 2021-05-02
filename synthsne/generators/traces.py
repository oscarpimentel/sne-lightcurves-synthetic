from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from flamingchoripan.datascience.xerror import XError
from . import lc_utils as lu
from . import exceptions as ex

###################################################################################################################################################

class Trace():
	def __init__(self):
		self.sne_models = []

	def append(self, sne_model):
		self.sne_models += [sne_model]

	def append_null(self):
		self.append(None)

	def get_fit_errors(self, lcobjb):
		self.fit_errors = []
		for k in range(0, len(self)):
			sne_model = self.sne_models[k]
			if sne_model is None:
				self.fit_errors += [np.infty]
			else:
				days, obs, obse = lu.extract_arrays(lcobjb)
				fit_error = sne_model.get_error(days, obs, obse)
				self.fit_errors += [fit_error]

	def sort(self):
		assert len(self.fit_errors)==len(self)
		idxs = np.argsort(self.fit_errors).tolist() # lower to higher
		self.sne_models = [self.sne_models[i] for i in idxs]
		self.fit_errors = [self.fit_errors[i] for i in idxs]

	def clip(self, n):
		assert n<=len(self)
		self.sne_models = self.sne_models[:n]
		self.fit_errors = self.fit_errors[:n]

	def get_valid_errors(self):
		return [self.fit_errors[k] for k in range(0, len(self)) if not self.sne_models[k] is None]

	def get_xerror(self):
		errors = self.get_valid_errors()
		return XError(errors)

	def get_xerror_k(self, k):
		assert k>=0 and k<len(self)
		sne_model = self.sne_models[k]
		if not sne_model is None and len(self)>0:
			return XError([self.fit_errors[k]])
		else:
			return XError(None)

	def has_corrects_samples(self):
		return any([not self.sne_models[k] is None for k in range(0, len(self))])

	def __len__(self):
		return len(self.sne_models)

	def __getitem__(self, k):
		return self.sne_models[k]