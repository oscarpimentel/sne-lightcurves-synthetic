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
		'f':(0, .95), # .5 .75 .9 1
		'trise':(1, 50),
		'tfall':(1, 130),
		#'s':(1e-1, 1e1),
		#'s':(1/3, 3),
	}
	return spm_bounds

def get_spm_random_sphere(spm_args, spm_bounds):
	new_spm_args = {}
	for p in spm_args.keys():
		spm_arg = spm_args[p]
		spm_bound = spm_bounds[p]
		r = abs(spm_bound[0]-spm_bound[1])*0.1 * np.random.randn()
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

def get_mcmc_priors(c, b):
	if c=='SNIa':
		if b=='g':
			A = stats.gamma(1.5614117006329225, 0, 0.2780335606877896) # &$\gammadist{1.56, 3.60, 0.00}$
			t0 = stats.norm(4.68021259907846, 5.359761365339392) # &$\normdist{4.68}{5.36}$
			gamma = stats.gamma(3.7788735568722034, 1, 6.561032502193379) # &$\gammadist{3.78, 0.15, 1.00}$
			trise = stats.gamma(2.8972409516753013, 1, 0.9278113914127201) # &$\gammadist{2.90, 1.08, 1.00}$
			tfall = stats.gamma(1.6747508057839176, 1, 29.606194112776457) # &$\gammadist{1.67, 0.03, 1.00}$
		if b=='r':
			A = stats.gamma(1.6760969814468585, 0, 0.2551834758835543) # &$\gammadist{1.68, 3.92, 0.00}$
			t0 = stats.norm(5.386420289895369, 6.529751400974657) # &$\normdist{5.39}{6.53}$
			gamma = stats.gamma(1.6668404171732187, 1, 15.890546847660556) # &$\gammadist{1.67, 0.06, 1.00}$
			trise = stats.gamma(2.33910157508384, 1, 1.3020925623075532) # &$\gammadist{2.34, 0.77, 1.00}$
			tfall = stats.gamma(1.9548436570015946, 1, 22.022028924137913) # &$\gammadist{1.95, 0.05, 1.00}$
	if c=='allSNII':
		if b=='g':
			A = stats.gamma(1.2518628975714086, 0, 0.31026586546086243) # &$\gammadist{1.25, 3.22, 0.00}$
			t0 = stats.norm(1.7205126861223679, 13.40563181633267) # &$\normdist{1.72}{13.41}$
			gamma = stats.gamma(1.804626566044503, 1, 23.994745096470698) # &$\gammadist{1.80, 0.04, 1.00}$
			trise = stats.gamma(0.6723120213719546, 1, 10.121672769850798) # &$\gammadist{0.67, 0.10, 1.00}$
			tfall = stats.gamma(1.652687414285144, 1, 33.576340017280195) # &$\gammadist{1.65, 0.03, 1.00}$
		if b=='r':
			A = stats.gamma(1.3581419676713224, 0, 0.30281794887113445) # &$\gammadist{1.36, 3.30, 0.00}$
			t0 = stats.norm(5.407048248979926, 21.678787967348047) # &$\normdist{5.41}{21.68}$
			gamma = stats.gamma(2.59217078774162, 1, 26.746000494495746) # &$\gammadist{2.59, 0.04, 1.00}$
			trise = stats.gamma(0.6025882020092925, 1, 18.930404697663683) # &$\gammadist{0.60, 0.05, 1.00}$
			tfall = stats.gamma(1.2083489233491747, 1, 42.56811740531283) # &$\gammadist{1.21, 0.02, 1.00}$
	if c=='SNIbc':
		if b=='g':
			A = stats.gamma(1.4171786008976806, 0, 0.26475684495936114) # &$\gammadist{1.42, 3.78, 0.00}$
			t0 = stats.norm(5.013815374794686, 7.596305870726241) # &$\normdist{5.01}{7.60}$
			gamma = stats.gamma(2.4938661110802807, 1, 9.319422321791306) # &$\gammadist{2.49, 0.11, 1.00}$
			trise = stats.gamma(1.0757430796077607, 1, 3.5846773835322123) # &$\gammadist{1.08, 0.28, 1.00}$
			tfall = stats.gamma(2.05055142287629, 1, 28.22636698095411) # &$\gammadist{2.05, 0.04, 1.00}$
		if b=='r':
			A = stats.gamma(1.4989121439768829, 0, 0.31587913914618043) # &$\gammadist{1.50, 3.17, 0.00}$
			t0 = stats.norm(4.1633889474881824, 7.191966539756747) # &$\normdist{4.16}{7.19}$
			gamma = stats.gamma(3.6350002545382822, 1, 8.68441314916002) # &$\gammadist{3.64, 0.12, 1.00}$
			trise = stats.gamma(1.535216918276577, 1, 2.368901114595958) # &$\gammadist{1.54, 0.42, 1.00}$
			tfall = stats.gamma(2.1536745147825003, 1, 24.21028477536784) # &$\gammadist{2.15, 0.04, 1.00}$
	if c=='SLSN':
		if b=='g':
			A = stats.gamma(1.7017334312329724, 0, 0.15526112240287424) # &$\gammadist{1.70, 6.44, 0.00}$
			t0 = stats.norm(12.179370234824278, 10.947301930740656) # &$\normdist{12.18}{10.95}$
			gamma = stats.gamma(7.277684868765533, 1, 11.0201860612152) # &$\gammadist{7.28, 0.09, 1.00}$
			trise = stats.gamma(2.1527610845370435, 1, 5.738711072312859) # &$\gammadist{2.15, 0.17, 1.00}$
			tfall = stats.gamma(1.375782414547162, 1, 48.51471124719929) # &$\gammadist{1.38, 0.02, 1.00}$
		if b=='r':
			A = stats.gamma(2.226251939393843, 0, 0.11179349603512159) # &$\gammadist{2.23, 8.95, 0.00}$
			t0 = stats.norm(16.05056519337847, 11.640622854991442) # &$\normdist{16.05}{11.64}$
			gamma = stats.gamma(5.037760826530633, 1, 15.805387680436022) # &$\gammadist{5.04, 0.06, 1.00}$
			trise = stats.gamma(2.893974461879907, 1, 4.210460512881061) # &$\gammadist{2.89, 0.24, 1.00}$
			tfall = stats.gamma(1.2893029458612935, 1, 42.83762704210187) # &$\gammadist{1.29, 0.02, 1.00}$

	f = None
	return A, t0, gamma, f, trise, tfall