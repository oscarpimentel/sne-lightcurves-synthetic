from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from . import lc_utils as lu
from scipy.optimize import fmin

###################################################################################################################################################

def get_pm_bounds(lcobjb, class_names,
	uses_new_bounds=True,
	min_required_points=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
	):
	days, obs, obs_error = lu.extract_arrays(lcobjb)

	### checks
	if len(days)<min_required_points:
		raise ex.TooShortCurveError()

	### utils
	min_flux = np.min(obs)
	max_flux = np.max(obs)
	mean_flux = np.mean(obs)
	first_flux = obs[0]
	day_max_flux = days[np.argmax(obs)]
	first_day = days.min()
	last_day = days.max()

	if not uses_new_bounds:
		pm_bounds = {
			'A':(max_flux / 3, max_flux * 3),
			't0':(-80, +80),
			'gamma':(1, 100),
			'f':(0, 1),
			'trise':(1, 100),
			'tfall':(1, 100),
			#'s':(1/3.-0.01, 1/3.+0.01),
			's':(1e-1, 2e1),
			#'s':(1e-1, 1e3),
			'g':(0, 1), # use with bernoulli
		}
		ret = {c:pm_bounds for c in class_names}
	else:
		pm_bounds = {
			'A':(max_flux / 3, max_flux * 3),
			't0':(day_max_flux-30, day_max_flux+10),
			#'gamma':(3, 100),
			'gamma':(5, 100),
			'f':(0, 1),
			'trise':(1, 20),
			'tfall':(5, 100),
			's':(1/3, 3),
			'g':(0, 1), # use with bernoulli
		}
		pm_bounds_slsn = {
			'A':(max_flux / 3, max_flux * 3),
			't0':(day_max_flux-100, day_max_flux+10),
			'gamma':(3, 150),
			'f':(0, 1),
			'trise':(1, 100),
			'tfall':(50, 300),
			's':(1/3, 3),
			'g':(0, 1), # use with bernoulli
		}
		ret = {c:pm_bounds for c in class_names}
		#ret.update({'SLSN':pm_bounds_slsn})
	return ret

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

def get_pm_times(func, inv_func, lcobjb, pm_args, pm_features, pm_bounds,
	min_obs_threshold=0,
	):
	t0 = pm_args['t0']
	first_day = lcobjb.days[0]
	last_day = lcobjb.days[-1]

	func_args = tuple([pm_args[pmf] for pmf in pm_features])
	tmax = fmin(inv_func, t0, func_args, disp=False)[0]

	### ti
	#search_range = tmax-pm_bounds['trise'][-1], tmax
	#search_range = tmax-pm_bounds['trise'][-1]*1, tmax
	search_range = min(tmax, first_day)-pm_bounds['trise'][-1], tmax
	ti = get_min_tfunc(search_range, func, func_args, min_obs_threshold)
	
	### tf
	#search_range = tmax-pm_bounds['tfall'][-1], tmax
	#search_range = tmax, tmax+pm_bounds['tfall'][-1]*3
	search_range = tmax, max(tmax, last_day)+pm_bounds['tfall'][-1]
	tf = get_min_tfunc(search_range, func, func_args, min_obs_threshold)

	assert tmax>=ti
	assert tf>=tmax
	pm_times = {
		'ti':ti,
		'tmax':tmax,
		'tf':tf,
	}
	return pm_times