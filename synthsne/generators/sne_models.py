from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep
from scipy.optimize import fmin
from . import exceptions as ex
from .functions import syn_sne_sfunc, inverse_syn_sne_sfunc, get_min_in_time_window

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
						spl = splrep(self.lcobjb.days, self.lcobjb.obs, w=self.lcobjb.obse**2+C_.REC_LOSS_EPS)
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

	def get_error(self, times, real_obs, real_obse,
		scale=C_.ERROR_SCALE,
		):
		syn_obs = self.evaluate(times)
		error = (real_obs-syn_obs)**2/(real_obse**2+C_.REC_LOSS_EPS)
		return error.mean()*scale

	def get_spm_times(self, min_obs_threshold, uses_estw,
		pre_tmax_offset=C_.PRE_TMAX_OFFSET,
		):
		first_day = self.lcobjb.days[0]
		last_day = self.lcobjb.days[-1]
		tmax_day = self.lcobjb.days[np.argmax(self.lcobjb.obs)]

		if uses_estw and not self.uses_interp():
			func_args = tuple([self.spm_args[p] for p in self.parameters])
			spm_tmax = fmin(self.inv_func, self.spm_args['t0'], func_args, disp=False)[0]
			first_day_prev = first_day-pre_tmax_offset
			spm_tmax_prev = spm_tmax-pre_tmax_offset

			### find peak and extend
			'''			
			if first_day<=spm_tmax_prev:
				ti = first_day
			else:
				ti_search_range = (spm_tmax_prev, spm_tmax)
				ti = get_min_in_time_window(ti_search_range, syn_sne_sfunc, func_args, min_obs_threshold)	'''

			if first_day<spm_tmax: # pre-preak observation exists
				ti = first_day
			else:
				ti = max(first_day_prev, spm_tmax)

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