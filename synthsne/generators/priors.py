from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
from . import lc_utils as lu
from . import exceptions as ex
import scipy.stats as stats
from scipy.optimize import curve_fit

###################################################################################################################################################

def get_spm_bounds(lcobjb, class_names):
	days, obs, obse = lu.extract_arrays(lcobjb)
	if len(days)<=1:
		return None

	### utils
	min_flux = np.min(obs)
	max_flux = np.max(obs)
	mean_flux = np.mean(obs)
	first_flux = obs[0]
	day_max_flux = days[np.argmax(obs)]
	first_day = days.min()
	last_day = days.max()

	spm_bounds = {
		'A':(max_flux/5, max_flux*5), # curve-wise
		't0':(first_day-10, day_max_flux+50), # curve-wise
		#'gamma':(3, 100),
		'gamma':(1, 120), # gamma is important
		'f':(0, 1),
		'trise':(1, 50),
		'tfall':(1, 130),
	}
	return spm_bounds

def get_spm_gaussian_sphere(spm_args, spm_bounds,
	k_std=1e-1,
	uniform_params=[],
	):
	d = {}
	for spm_p in spm_bounds.keys():
		if spm_p in uniform_params:
			d[spm_p] = stats.uniform(min(spm_bounds[spm_p]), max(spm_bounds[spm_p]))
		else:
			std = (max(spm_bounds[spm_p])-min(spm_bounds[spm_p]))*k_std
			d[spm_p] = stats.truncnorm(min(spm_bounds[spm_p]), max(spm_bounds[spm_p]), spm_args[spm_p], std)

	return d

def get_spm_uniform_box(spm_bounds):
	return {spm_p:stats.uniform(min(spm_bounds[spm_p]), max(spm_bounds[spm_p])) for spm_p in spm_bounds.keys()}

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
	f_guess = 0.5
	
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