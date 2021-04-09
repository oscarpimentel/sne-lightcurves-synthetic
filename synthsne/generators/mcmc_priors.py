from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from . import lc_utils as lu
from . import exceptions as ex
import scipy.stats as stats

###################################################################################################################################################

class MCMCPrior():
	def __init__(self, raw_samples, scipy_dist_name, floc, fscale):
		self.raw_samples = raw_samples.copy()
		self.scipy_dist_name = scipy_dist_name
		self.floc = floc
		self.fscale = fscale
		self.reset()

	def reset(self):
		self.samples = self.filter(self.raw_samples)
		self.fit()

	def filter(self, raw_samples):
		return raw_samples.copy()

	def __len__(self):
		return len(self.samples)

	def fit(self):
		self.distr = getattr(stats, self.scipy_dist_name)
		self.dist_params = self.distr.fit(self.samples, floc=self.floc, fscale=self.fscale)

	def sample(self, n):
		return self.distr.rvs(*self.dist_params, size=n)

	def pdf(self, x):
		return self.distr.pdf(x, *self.dist_params)

class GammaP(MCMCPrior):
	def __init__(self, _raw_samples_,
		floc=1,
		eps=C_.EPS,
		):
		#raw_samples = np.clip(_raw_samples_, floc+eps, None)
		raw_samples = np.array(_raw_samples_)
		raw_samples = raw_samples[raw_samples>=floc*1.1]
		raw_samples = raw_samples.tolist()
		#p = p[(p>floc*1.05) & p>floc*1.05)]s
		
		#fscale = None if dist_name in ['norm', 'gamma'] else 1
		scipy_dist_name = 'gamma'
		fscale = None
		super().__init__(raw_samples, scipy_dist_name, floc, fscale)
		
	def __repr__(self):
		alpha = self.dist_params[0]
		mu = self.dist_params[1]
		scale = self.dist_params[2]
		beta = 1/scale
		txt = '$\\gammadist{'+f'{alpha:.2f}, {beta:.2f}, {mu:.2f}'+'}$'
		return txt

class NormalP(MCMCPrior):
	def __init__(self, raw_samples):
		scipy_dist_name = 'norm'
		floc = None
		fscale = None
		super().__init__(raw_samples, scipy_dist_name, floc, fscale)

	def __repr__(self):
		mu = self.dist_params[0]
		scale = self.dist_params[1]
		txt = '$\\normdist{'+f'{mu:.2f}'+'}{'+f'{scale:.2f}'+'}$'
		return txt

class UniformP(MCMCPrior):
	def __init__(self):
		self.sne_model_l = []
		self.spm_bounds_l = []
		self.fit_errors = []
		self.correct_fit_tags = []

	def add(self, sne_model, spm_bounds, correct_fit_tag):
		self.sne_model_l.append(sne_model)