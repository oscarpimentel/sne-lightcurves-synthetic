from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep
from scipy.optimize import fmin
from . import exceptions as ex
from .functions import syn_sne_sfunc, inverse_syn_sne_sfunc, get_min_in_time_window

REC_LOSS_EPS = _C.REC_LOSS_EPS
REC_LOSS_K = _C.REC_LOSS_K
ERROR_SCALE = _C.ERROR_SCALE
PRE_TMAX_OFFSET = _C.PRE_TMAX_OFFSET

###################################################################################################################################################

class SNeModel():
	def __init__(self, lcobjb, spm_type, spm_bounds, spm_args):
		self.lcobjb = lcobjb.copy()
		self.spm_type = spm_type
		self.spm_bounds = spm_bounds
		self.spm_args = spm_args
		self.reset()

	def reset(self):
		self.parameters = ['A', 't0', 'gamma', 'f', 'trise', 'tfall']
		self.func = syn_sne_sfunc
		self.inv_func = inverse_syn_sne_sfunc

	def uses_interp(self):
		return self.spm_type in ['linear', 'bspline']

	def evaluate(self, times):
		if self.uses_interp():
			if len(self.lcobjb)>1:
				if self.spm_type=='bspline':
					try:
						sigma = REC_LOSS_EPS+REC_LOSS_K*(self.lcobjb.obse**2)
						spl = splrep(self.lcobjb.days, self.lcobjb.obs, w=sigma)
						parametric_obs = splev(times, spl)
					except TypeError:
						raise ex.BSplineError()
				elif self.spm_type=='linear':
					interp = interp1d(self.lcobjb.days, self.lcobjb.obs, kind='linear', fill_value='extrapolate')
					parametric_obs = interp(times)
			else:
				parametric_obs = np.array([self.lcobjb.obs[0]])
		else:
			parametric_obs = self.func(times, *[self.spm_args[p] for p in self.parameters])
		return parametric_obs

	def evaluate_inv(self, times):
		if self.uses_interp():
			raise Exception('not implemented')
		else:
			return self.inv_func(times, *[self.spm_args for p in self.parameters])

	def get_error(self, times, real_obs, real_obse):
		syn_obs = self.evaluate(times)
		sigma = REC_LOSS_EPS+REC_LOSS_K*(real_obse**2)
		error = (real_obs-syn_obs)**2/sigma
		return error.mean()*ERROR_SCALE

	def get_spm_times(self, min_obs_threshold, uses_estw,
		pre_tmax_offset=PRE_TMAX_OFFSET,
		):
		first_day = self.lcobjb.days[0]
		last_day = self.lcobjb.days[-1]
		tmax_day = self.lcobjb.days[np.argmax(self.lcobjb.obs)]
		if uses_estw and not self.uses_interp():
			func_args = tuple([self.spm_args[p] for p in self.parameters])
			spm_tmax = fmin(self.inv_func, self.spm_args['t0'], func_args, disp=False)[0]
			if first_day<spm_tmax: # pre-preak observation exists
				ti = first_day
			else:
				ti = first_day-pre_tmax_offset
			# ti = min(first_day, spm_tmax-pre_tmax_offset)
			spm_times = {
				'ti':ti,
				'spm_tmax':spm_tmax,
				'tmax_day':tmax_day,
				'tf':last_day,
			}
		else:
			spm_times = {
				'ti':first_day,
				'spm_tmax':None,
				'tmax_day':tmax_day,
				'tf':last_day,
			}
		#assert tmax>=ti
		#assert tf>=tmax
		assert spm_times['tf']>=spm_times['ti']
		self.spm_times = spm_times
		return self.spm_times