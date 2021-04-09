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

def get_mcmc_priors(c, b):
	if c=='SNIa':
		if b=='g':
			A = stats.gamma(1.5498733309341117, 0, 0.2795232486368346) # &$\gammadist{1.55, 3.58, 0.00}$
			t0 = stats.norm(4.654260062597296, 5.436825845952486) # &$\normdist{4.65}{5.44}$
			gamma = stats.gamma(3.502686986983534, 1, 7.054436975038102) # &$\gammadist{3.50, 0.14, 1.00}$
			trise = stats.gamma(2.740654390615995, 1, 0.9854416687941565) # &$\gammadist{2.74, 1.01, 1.00}$
			tfall = stats.gamma(1.619485315155995, 1, 30.837816643175234) # &$\gammadist{1.62, 0.03, 1.00}$
		if b=='r':
			A = stats.gamma(1.673914524938186, 0, 0.2581824680585143) # &$\gammadist{1.67, 3.87, 0.00}$
			t0 = stats.norm(5.79461542912225, 6.567483525739507) # &$\normdist{5.79}{6.57}$
			gamma = stats.gamma(1.467388807313481, 1, 17.221554858060845) # &$\gammadist{1.47, 0.06, 1.00}$
			trise = stats.gamma(2.4849597320879715, 1, 1.24954372631457) # &$\gammadist{2.48, 0.80, 1.00}$
			tfall = stats.gamma(1.8417869642657363, 1, 23.19300017081047) # &$\gammadist{1.84, 0.04, 1.00}$
	if c=='allSNII':
		if b=='g':
			A = stats.gamma(1.2444905162819877, 0, 0.31492230898391327) # &$\gammadist{1.24, 3.18, 0.00}$
			t0 = stats.norm(2.0887621565208003, 13.731109078078958) # &$\normdist{2.09}{13.73}$
			gamma = stats.gamma(1.7114679935147428, 1, 24.576005691595146) # &$\gammadist{1.71, 0.04, 1.00}$
			trise = stats.gamma(0.6598839672021936, 1, 10.736309218257032) # &$\gammadist{0.66, 0.09, 1.00}$
			tfall = stats.gamma(1.676562032446677, 1, 33.27324764118627) # &$\gammadist{1.68, 0.03, 1.00}$
		if b=='r':
			A = stats.gamma(1.3159645855437898, 0, 0.3179264864871128) # &$\gammadist{1.32, 3.15, 0.00}$
			t0 = stats.norm(5.524624062494766, 21.64169903458385) # &$\normdist{5.52}{21.64}$
			gamma = stats.gamma(2.527520975891993, 1, 26.903312763861393) # &$\gammadist{2.53, 0.04, 1.00}$
			trise = stats.gamma(0.6104776495363632, 1, 18.654929727133954) # &$\gammadist{0.61, 0.05, 1.00}$
			tfall = stats.gamma(1.207395441949931, 1, 41.946606182188944) # &$\gammadist{1.21, 0.02, 1.00}$
	if c=='SNIbc':
		if b=='g':
			A = stats.gamma(1.406577988719368, 0, 0.2662005410471651) # &$\gammadist{1.41, 3.76, 0.00}$
			t0 = stats.norm(4.910236805410485, 7.68466092078336) # &$\normdist{4.91}{7.68}$
			gamma = stats.gamma(2.3273357239812777, 1, 9.97399579023476) # &$\gammadist{2.33, 0.10, 1.00}$
			trise = stats.gamma(1.080831374444715, 1, 3.4745486186272085) # &$\gammadist{1.08, 0.29, 1.00}$
			tfall = stats.gamma(1.9339205761826777, 1, 29.475328275082475) # &$\gammadist{1.93, 0.03, 1.00}$
		if b=='r':
			A = stats.gamma(1.5525978471378918, 0, 0.30483056637239136) # &$\gammadist{1.55, 3.28, 0.00}$
			t0 = stats.norm(4.091090684497759, 8.083606643707025) # &$\normdist{4.09}{8.08}$
			gamma = stats.gamma(3.999402796224063, 1, 7.6943312629199285) # &$\gammadist{4.00, 0.13, 1.00}$
			trise = stats.gamma(1.3962948483219, 1, 2.7180944426289737) # &$\gammadist{1.40, 0.37, 1.00}$
			tfall = stats.gamma(2.1625890347372443, 1, 24.721536732496343) # &$\gammadist{2.16, 0.04, 1.00}$
	if c=='SLSN':
		if b=='g':
			A = stats.gamma(1.6859701143629, 0, 0.15705166449868238) # &$\gammadist{1.69, 6.37, 0.00}$
			t0 = stats.norm(11.83229567321145, 11.149557666998158) # &$\normdist{11.83}{11.15}$
			gamma = stats.gamma(7.530448564243066, 1, 10.480655589738385) # &$\gammadist{7.53, 0.10, 1.00}$
			trise = stats.gamma(2.216485028054478, 1, 5.555394302571985) # &$\gammadist{2.22, 0.18, 1.00}$
			tfall = stats.gamma(1.3185990703985566, 1, 49.678181187359606) # &$\gammadist{1.32, 0.02, 1.00}$
		if b=='r':
			A = stats.gamma(2.3634054610891337, 0, 0.10630583808935297) # &$\gammadist{2.36, 9.41, 0.00}$
			t0 = stats.norm(18.676855369691214, 12.759839089827748) # &$\normdist{18.68}{12.76}$
			gamma = stats.gamma(4.676176565846804, 1, 16.201377578124905) # &$\gammadist{4.68, 0.06, 1.00}$
			trise = stats.gamma(3.1933551565192566, 1, 3.913740791705764) # &$\gammadist{3.19, 0.26, 1.00}$
			tfall = stats.gamma(1.376621540026383, 1, 47.43268548365236) # &$\gammadist{1.38, 0.02, 1.00}$

	f = gamma = stats.uniform(0, 1)
	return A, t0, gamma, f, trise, tfall