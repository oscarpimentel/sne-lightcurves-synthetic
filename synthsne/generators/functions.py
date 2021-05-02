from __future__ import print_function
from __future__ import division
from . import C_

from numba import jit
import numpy as np

###################################################################################################################################################

def get_min_in_time_window(search_range, func, func_args,
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

@jit(nopython=True)
def sgm(x, x0, s):
	return 1./(1. + np.exp(-s*(x-x0)))

@jit(nopython=True)
def syn_sne_sfunc(t, A, t0, gamma, f, trise, tfall):
	s = 1/5
	g = sgm(t, gamma+t0, s)
	early = 1.0*(A*(1 - (f*(t-t0)/gamma))   /   (1. + np.exp(-(t-t0)/trise)))
	late = 1.0*(A*(1-f)*np.exp(-(t-(gamma+t0))/tfall)   /   (1. + np.exp(-(t-t0)/trise)))
	flux = (1.-g)*early + g*late
	return flux

@jit(nopython=True)
def inverse_syn_sne_sfunc(t, A, t0, gamma, f, trise, tfall):
	return -syn_sne_sfunc(t, A, t0, gamma, f, trise, tfall)

###################################################################################################################################################

@jit(nopython=True)
def log_prior(A_pdf, t0_pdf, gamma_pdf, f_pdf, trise_pdf, tfall_pdf,
	eps=C_.EPS,
	):
	p = 0
	if A_pdf>0:
		p += np.log(A_pdf) 
	else:
		return -np.inf

	if t0_pdf>0:
		p += np.log(t0_pdf)
	else:
		return -np.inf
	
	if gamma_pdf>0:
		p += np.log(gamma_pdf)
	else:
		return -np.inf
	
	if f_pdf>0:
		p += np.log(f_pdf)
	else:
		return -np.inf

	if trise_pdf>0:
		p += np.log(trise_pdf)
	else:
		return -np.inf
	
	if tfall_pdf>0:
		p += np.log(tfall_pdf)
	else:
		return -np.inf

	return p

@jit(nopython=True)
def log_likelihood(spm_obs, days, obs, obse):
	#sigma = obse**2+C_.EPS#+C_.REC_LOSS_EPS
	sigma = obse**2+C_.REC_LOSS_EPS
	return -0.5 * np.sum((obs - spm_obs)**2/sigma + np.log(sigma))

#@jit(nopython=True)
def log_probability(theta, d_theta, func, days, obs, obse):
	A, t0, gamma, f, trise, tfall = theta
	A_pdf, t0_pdf, gamma_pdf, f_pdf, trise_pdf, tfall_pdf = [distr.pdf(x) for x,distr in zip(theta, d_theta)]
	lp = log_prior(A_pdf, t0_pdf, gamma_pdf, f_pdf, trise_pdf, tfall_pdf)
	if not np.isfinite(lp): # not finite
		return -np.inf # big negative number
	spm_obs = func(days, *theta)
	log_probability = lp + log_likelihood(spm_obs, days, obs, obse)
	return log_probability