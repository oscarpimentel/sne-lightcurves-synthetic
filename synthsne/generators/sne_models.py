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
		self.uses_interp = self.spm_type in ['linear', 'bspline']
		self.func = syn_sne_sfunc
		self.inv_func = inverse_syn_sne_sfunc

	def evaluate(self, times):
		if self.uses_interp:
			if len(self.lcobjb)>1:
				if self.spm_type=='bspline':
					try:
						spl = splrep(self.lcobjb.days, self.lcobjb.obs, w=self.lcobjb.obse**2)
						parametric_obs = splev(times, spl)
					except TypeError:
						self.spm_type = 'linear'

				if self.spm_type=='linear':
					interp = interp1d(self.lcobjb.days, self.lcobjb.obs, kind='linear', fill_value='extrapolate')
					parametric_obs = interp(times)
			else:
				parametric_obs = np.array([self.lcobjb.obs[0]])
		else:
			parametric_obs = self.func(times, *[self.spm_args[p] for p in self.parameters])
		return parametric_obs

	def evaluate_inv(self, times):
		if self.uses_interp:
			raise Exception('not implemented')
		else:
			return self.inv_func(times, *[self.spm_args for p in self.parameters])

	def get_error(self, times, real_obs, real_obse,
		scale=C_.ERROR_SCALE,
		):
		syn_obs = self.evaluate(times)
		error = (real_obs-syn_obs)**2/(real_obse**2+.1)
		return error.mean()*scale

	def get_spm_times(self, min_obs_threshold, uses_estw,
		pre_tmax_offset=15, # 0 1 5 10 20
		):
		first_day = self.lcobjb.days[0]
		last_day = self.lcobjb.days[-1]
		tmax_day = self.lcobjb.days[np.argmax(self.lcobjb.obs)]

		if uses_estw and not self.uses_interp:
			func_args = tuple([self.spm_args[p] for p in self.parameters])
			t0 = self.spm_args['t0']
			spm_tmax = fmin(self.inv_func, t0, func_args, disp=False)[0]

			### ti
			if first_day>spm_tmax-pre_tmax_offset:
				ti_search_range = (spm_tmax-pre_tmax_offset, spm_tmax)
				ti = get_min_in_time_window(ti_search_range, syn_sne_sfunc, func_args, min_obs_threshold)
			else:
				ti = first_day

			### tf
			#tf_offset = 5 # 0 1 5
			#tf_search_range = tmax, max(tmax, last_day)+self.spm_args['tfall']*0.2
			#tf = get_min_in_time_window(tf_search_range, syn_sne_sfunc, func_args, min_obs_threshold)
			#tf = max(tmax, last_day)
			tf = last_day

			spm_times = {
				'ti':ti,
				'spm_tmax':spm_tmax,
				'tmax_day':tmax_day,
				'tf':tf,
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