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

	def clean(self):
		self.raw_samples = None
		self.samples = None
		return self

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
		floc=1.,
		eps=C_.EPS,
		**kwargs):
		#raw_samples = np.clip(_raw_samples_, floc+eps, None)
		raw_samples = np.array(_raw_samples_)
		raw_samples = raw_samples[raw_samples>=floc*1.05]
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
		txt = '$\\text{Gamma}\\left('+f'{alpha:.3f}, {beta:.3f}, {mu:.3f}'+'\\right)$'
		return txt

class NormalP(MCMCPrior):
	def __init__(self, raw_samples,
		**kwargs):
		scipy_dist_name = 'norm'
		floc = None
		fscale = None
		super().__init__(raw_samples, scipy_dist_name, floc, fscale)

	def __repr__(self):
		mu = self.dist_params[0]
		scale = self.dist_params[1]
		txt = '$\\text{Normal}\\left('+f'{mu:.3f}, {scale:.3f}'+'\\right)$'
		return txt

class UniformP(MCMCPrior):
	def __init__(self, raw_samples,
		**kwargs):
		floc = None
		fscale = None
		scipy_dist_name = 'uniform'
		super().__init__(raw_samples, scipy_dist_name, floc, fscale)
		
	def __repr__(self):
		loc = self.dist_params[0]
		scale = self.dist_params[1]
		b = loc+scale
		txt = '$\\text{Unif}\\left('+f'{loc:.3f}, {b:.3f}'+'\\right)$'
		return txt