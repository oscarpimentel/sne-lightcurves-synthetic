from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from . import lc_utils as lu
from . import exceptions as ex
import scipy.stats as stats

###################################################################################################################################################

def get_spm_bounds(lcobjb, class_names,
	uses_new_bounds=True,
	min_required_points=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
	min_required_duration=C_.MIN_DUR_LIGHTCURVE_TO_PMFIT, # min duration to even try a curve fit
	):
	days, obs, obse = lu.extract_arrays(lcobjb)

	### checks
	if len(lcobjb)<min_required_points or lcobjb.get_days_duration()<min_required_duration:
		raise ex.TooShortCurveError()
	if lcobjb.get_snr()<=C_.MIN_SNR:
		#raise ex.TooFaintCurveError()
		pass

	### utils
	min_flux = np.min(obs)
	max_flux = np.max(obs)
	mean_flux = np.mean(obs)
	first_flux = obs[0]
	day_max_flux = days[np.argmax(obs)]
	first_day = days.min()
	last_day = days.max()

	spm_bounds = {
		'A':(max_flux / 3, max_flux * 3), # curve-wise
		't0':(first_day-50, day_max_flux+50), # curve-wise
		#'gamma':(3, 100),
		'gamma':(1, 120), # gamma is important
		'f':(0, .99), # .5 .75 .99 1
		'trise':(1, 50),
		'tfall':(1, 130),
	}
	return spm_bounds

def get_spm_random_sphere(spm_args, spm_bounds):
	new_spm_args = {}
	for p in spm_args.keys():
		spm_arg = spm_args[p]
		spm_bound = spm_bounds[p]
		r = abs(spm_bound[0]-spm_bound[1])*0.01 * np.random.randn()
		new_spm_arg = spm_arg + r
		new_spm_args[p] = np.clip(new_spm_arg, *spm_bound)
	return new_spm_args

def get_p0(lcobjb, spm_bounds):
	days, obs, obse = lu.extract_arrays(lcobjb)

	### utils
	new_days = days-days[0]
	min_flux = np.min(obs)
	max_flux = np.max(obs)
	mean_flux = np.mean(obs)
	first_flux = obs[0]
	max_flux_day = new_days[np.argmax(obs)]
	first_day = new_days.min()
	last_day = new_days.max()
	frac_r = 0.2

	### A
	A_guess = 1.2*max_flux

	### t0
	t0_guess = max_flux_day
	
	### gamma
	mask = obs >= max_flux / 3. #np.percentile(obs, 33)
	gamma_guess = new_days[mask].max() - new_days[mask].min() if mask.sum() > 0 else spm_bounds['gamma'][0]

	### f
	f_guess = (spm_bounds['f'][0]+spm_bounds['f'][-1])/2
	
	### trise
	trise_guess = (max_flux_day - first_day) / 2.
	
	### tfall
	tfall_guess = 40.

	### set
	p0 = {
		'A':np.clip(A_guess, spm_bounds['A'][0], spm_bounds['A'][-1]),
		't0':np.clip(t0_guess, spm_bounds['t0'][0], spm_bounds['t0'][-1]),
		'gamma':np.clip(gamma_guess, spm_bounds['gamma'][0], spm_bounds['gamma'][-1]),
		'f':np.clip(gamma_guess, spm_bounds['f'][0], spm_bounds['f'][-1]),
		'trise':np.clip(trise_guess, spm_bounds['trise'][0], spm_bounds['trise'][-1]),
		'tfall':np.clip(tfall_guess, spm_bounds['tfall'][0], spm_bounds['tfall'][-1]),
	}
	return p0