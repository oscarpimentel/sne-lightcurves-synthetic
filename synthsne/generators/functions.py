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
	lp_A = np.log(A_pdf+eps)
	lp_t0 = np.log(t0_pdf+eps)
	lp_gamma = np.log(gamma_pdf+eps)
	lp_f = np.log(f_pdf+eps)
	lp_trise = np.log(trise_pdf+eps)
	lp_tfall = np.log(tfall_pdf+eps)
	return lp_A + lp_t0 + lp_gamma + lp_f + lp_trise + lp_tfall

@jit(nopython=True)
def log_likelihood(spm_obs, days, obs, obse,
	eps=C_.EPS,
	):
	sigma2 = (obse*1)**2+eps
	return -0.5 * np.sum((obs - spm_obs)**2/sigma2 + np.log(sigma2))

#@jit(nopython=True)
def log_probability(theta, d_theta, func, days, obs, obse,
	eps=C_.EPS,
	):
	A, t0, gamma, f, trise, tfall = theta
	A_pdf, t0_pdf, gamma_pdf, f_pdf, trise_pdf, tfall_pdf = [p.pdf(x) for x,p in zip(theta,d_theta)]
	lp = log_prior(A_pdf, t0_pdf, gamma_pdf, f_pdf, trise_pdf, tfall_pdf)
	if not np.isfinite(lp):
		return -1/eps
	spm_obs = func(days, *theta)
	log_probability = lp + log_likelihood(spm_obs, days, obs, obse)
	#print(log_probability)
	return log_probability