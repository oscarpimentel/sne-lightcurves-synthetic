from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep
from scipy.optimize import fmin
from . import exceptions as ex

###################################################################################################################################################

def sgm(x, x0, s):
	return 1/(1 + np.exp(-s*(x-x0)))

'''
def syn_sne_func(t, A, t0, gamma, f, trise, tfall):
	s = 1/3
	#s = 1
	#s = 10

	g = sgm(t, gamma+t0, s)
	early = 1.0*(A*(1 - (f*(t-t0)/gamma))   /   (1 + np.exp(-(t-t0)/trise)))
	late = 1.0*(A*(1-f)*np.exp(-(t-(gamma+t0))/tfall)   /   (1 + np.exp(-(t-t0)/trise)))
	flux = (1-g)*early + g*late
	return flux

def inverse_syn_sne_func(t, A, t0, gamma, f, trise, tfall):
	return -syn_sne_func(t, A, t0, gamma, f, trise, tfall)

'''


def syn_sne_sfunc(t, A, t0, gamma, f, trise, tfall,
	#s=1/3,
	):
	s = 1/3
	g = sgm(t, gamma+t0, s)
	early = 1.0*(A*(1 - (f*(t-t0)/gamma))   /   (1 + np.exp(-(t-t0)/trise)))
	late = 1.0*(A*(1-f)*np.exp(-(t-(gamma+t0))/tfall)   /   (1 + np.exp(-(t-t0)/trise)))
	flux = (1-g)*early + g*late
	return flux

def inverse_syn_sne_sfunc(t, A, t0, gamma, f, trise, tfall,
	#s=1/3,
	):
	return -syn_sne_sfunc(t, A, t0, gamma, f, trise, tfall)

def get_min_tfunc(search_range, func, func_args,
	min_obs_threshold=0,
	n=1e4,
	):
	lin_times = np.linspace(*search_range, int(n))
	func_v = func(lin_times, *func_args)
	valid_indexs = np.where(func_v>min_obs_threshold)[0]
	lin_times = lin_times[valid_indexs]
	func_v = func_v[valid_indexs]
	return lin_times[np.argmin(func_v)]

###################################################################################################################################################

class SNeModel():
	def __init__(self, lcobjb, pm_args,
		kind=None, # linear, bspline
		):
		#self.parameters = ['A', 't0', 'gamma', 'f', 'trise', 'tfall', 's']
		self.parameters = ['A', 't0', 'gamma', 'f', 'trise', 'tfall']
		self.lcobjb = lcobjb.copy()
		self.pm_args = {} if pm_args is None else pm_args.copy()
		self.kind = kind
		self.uses_interp = not self.kind is None

		self.func = syn_sne_sfunc
		self.inv_func = inverse_syn_sne_sfunc

	def evaluate(self, times):
		if self.uses_interp:
			try:
				if self.kind=='linear':
					interp = interp1d(self.lcobjb.days, self.lcobjb.obs, kind='linear', fill_value='extrapolate')
					obs = interp(times)

				elif self.kind=='bspline':
					spl = splrep(self.lcobjb.days, self.lcobjb.obs, w=self.lcobjb.obse**2)
					obs = splev(times, spl)
			except:
				raise ex.InterpError()

		else:
			obs = self.func(times, *[self.pm_args[p] for p in self.parameters])

		#obs = np.clip(obs, , )
		return obs

	def evaluate_inv(self, times):
		if self.uses_interp:
			raise Exception('not implemented')
		else:
			return self.inv_func(times, *[self.pm_args for p in self.parameters])

	def get_error(self, times, real_obs, real_obse,
		scale=C_.ERROR_SCALE,
		):
		syn_obs = self.evaluate(times)
		error = (real_obs-syn_obs)**2/(real_obse**2)
		return error.mean()*scale

	def get_pm_times(self, min_obs_threshold):
		first_day = self.lcobjb.days[0]
		last_day = self.lcobjb.days[-1]
		tmax_day = self.lcobjb.days[np.argmax(self.lcobjb.obs)]

		if self.uses_interp:
			pm_times = {
				'ti':first_day,
				'tmax':tmax_day,
				'tf':last_day,
			}
			ti_search_range = None
			tf_search_range = None
		else:
			func_args = tuple([self.pm_args[p] for p in self.parameters])
			t0 = self.pm_args['t0']
			tmax = fmin(self.inv_func, t0, func_args, disp=False)[0]

			### ti
			ti_search_range = min(tmax, first_day)-self.pm_args['trise']*0.9, tmax
			ti = get_min_tfunc(ti_search_range, syn_sne_sfunc, func_args, min_obs_threshold)
			
			### tf
			tf_search_range = tmax, max(tmax, last_day)+self.pm_args['tfall']*0.2
			tf = get_min_tfunc(tf_search_range, syn_sne_sfunc, func_args, min_obs_threshold)

			assert tmax>=ti
			assert tf>=tmax
			pm_times = {
				'ti':ti,
				'tmax':tmax,
				'tf':tf,
			}
		self.pm_times = pm_times
		return self.pm_times, ti_search_range, tf_search_range