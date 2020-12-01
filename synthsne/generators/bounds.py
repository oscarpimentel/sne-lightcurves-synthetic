from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from . import lc_utils as lu
from . import exceptions as ex

###################################################################################################################################################

def get_pm_bounds(lcobjb, class_names,
	uses_new_bounds=True,
	min_required_points=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
	min_required_duration=C_.MIN_DUR_LIGHTCURVE_TO_PMFIT, # min duration to even try a curve fit
	):
	days, obs, obs_error = lu.extract_arrays(lcobjb)

	### checks
	if len(lcobjb)<min_required_points or lcobjb.get_days_duration()<min_required_duration or lcobjb.get_snr()<=C_.MIN_SNR:
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
		}
		ret = {c:pm_bounds for c in class_names}
	else: # new bounds
		pm_bounds = {
			'A':(max_flux / 3, max_flux * 3),
			't0':(day_max_flux-30, day_max_flux+10),
			#'gamma':(3, 100),
			'gamma':(5, 100), # gamma is important important
			'f':(0, 1),
			'trise':(1, 18),
			'tfall':(5, 100),
			#'s':(1e-1, 1e1),
			's':(1/3, 3),
		}
		'''
		pm_bounds_slsn = {
			'A':(max_flux / 3, max_flux * 3),
			't0':(day_max_flux-100, day_max_flux+10),
			'gamma':(3, 150),
			'f':(0, 1),
			'trise':(1, 100),
			'tfall':(50, 300),
			's':(1/3, 3),
		}
		'''
		ret = {c:pm_bounds for c in class_names}
		#ret.update({'SLSN':pm_bounds_slsn})
	return ret