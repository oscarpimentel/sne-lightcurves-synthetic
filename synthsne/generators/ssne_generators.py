from __future__ import print_function
from __future__ import division
from . import _C

import numpy as np
import random
import emcee
from scipy.optimize import curve_fit
from . import exceptions as ex
from . import lc_utils as lu
from .sne_models import SNeModel
from . import priors as priors
from nested_dict import nested_dict
from .ssne_generator import SynSNeGenerator
from . import functions as fs
from .traces import Trace
import scipy.stats as stats

###################################################################################################################################################

def override(func): return func # tricky
class SynSNeGeneratorLinear(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
		n_trace_samples=_C.N_TRACE_SAMPLES,
		max_fit_error:float=_C.MAX_FIT_ERROR,
		std_scale:float=_C.OBSE_STD_SCALE,
		min_cadence_days:float=_C.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=_C.MIN_POINTS_LIGHTCURVE_DEFINITION,
		hours_noise_amp:float=_C.HOURS_NOISE_AMP,
		ignored=False,
		**kwargs):
		super().__init__(lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
			n_trace_samples,
			max_fit_error,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			hours_noise_amp,
			ignored,
			)

	@override
	def get_spm_trace_b(self, b, n):
		trace = Trace()
		for _ in range(max(n, self.n_trace_samples)):
			lcobjb = self.lcobj.get_b(b).copy()
			if len(lcobjb)>0:
				spm_bounds = priors.get_spm_bounds(lcobjb, self.class_names)
				sne_model = SNeModel(lcobjb, 'linear', spm_bounds, None)
				trace.append(sne_model)
			else:
				trace.append_null()
		return trace

###################################################################################################################################################

class SynSNeGeneratorBSpline(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
		n_trace_samples=_C.N_TRACE_SAMPLES,
		max_fit_error:float=_C.MAX_FIT_ERROR,
		std_scale:float=_C.OBSE_STD_SCALE,
		min_cadence_days:float=_C.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=_C.MIN_POINTS_LIGHTCURVE_DEFINITION,
		hours_noise_amp:float=_C.HOURS_NOISE_AMP,
		ignored=False,
		**kwargs):
		super().__init__(lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
			n_trace_samples,
			max_fit_error,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			hours_noise_amp,
			ignored,
			)

	@override
	def get_spm_trace_b(self, b, n):
		trace = Trace()
		for _ in range(max(n, self.n_trace_samples)):
			lcobjb = self.lcobj.get_b(b).copy()
			if len(lcobjb)>0:
				spm_bounds = priors.get_spm_bounds(lcobjb, self.class_names)
				try:
					sne_model = SNeModel(lcobjb, 'bspline', spm_bounds, None)
					sne_model.evaluate(lcobjb.days[0]) # trigger exception
				except ex.BSplineError:
					sne_model = SNeModel(lcobjb, 'linear', spm_bounds, None)

				trace.append(sne_model)
			else:
				trace.append_null()
		return trace

###################################################################################################################################################

class SynSNeGeneratorMLE(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
		n_trace_samples=_C.N_TRACE_SAMPLES,
		max_fit_error:float=_C.MAX_FIT_ERROR,
		std_scale:float=_C.OBSE_STD_SCALE,
		min_cadence_days:float=_C.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=_C.MIN_POINTS_LIGHTCURVE_DEFINITION,
		hours_noise_amp:float=_C.HOURS_NOISE_AMP,
		ignored=False,
		**kwargs):
		super().__init__(lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
			n_trace_samples,
			max_fit_error,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			hours_noise_amp,
			ignored,
			)

	def get_curvefit_spm_args(self, lcobjb, spm_bounds, _f):
		if len(lcobjb)<_C.MIN_POINTS_LIGHTCURVE_FOR_SPMFIT or lcobjb.get_days_duration()<_C.MIN_DUR_LIGHTCURVE_FOR_SPMFIT:
			raise ex.CurveFitError()
		days, obs, obse = lu.extract_arrays(lcobjb)
		p0 = priors.get_p0(lcobjb, spm_bounds)

		### solve nans
		invalid_indexs = (obs == np.infty) | (obs == -np.infty) | np.isnan(obs)
		obs[invalid_indexs] = 0 # as a patch, use 0
		obse[invalid_indexs] = 1/_C.EPS # as a patch, use a big obs error to null obs

		### bounds
		fit_kwargs = {
			#'method':'dogbox', # lm trf dogbox
			#'absolute_sigma':True,
			#'maxfev':1e6,
			'check_finite':True,
			'bounds':([spm_bounds[p][0] for p in spm_bounds.keys()], [spm_bounds[p][-1] for p in spm_bounds.keys()]),
			'ftol':p0['A']/20., # A_guess
			#'ftol':_C.CURVE_FIT_FTOL,
			'sigma':_C.RE_CLOSS_EPS+_C.RE_CLOSS_K*(obse**2),
		}

		### fitting
		try:
			p0_ = [p0[p] for p in spm_bounds.keys()]
			popt, pcov = curve_fit(_f, days, obs, p0=p0_, **fit_kwargs)
		
		except (ValueError, RuntimeError):
			raise ex.CurveFitError()

		spm_args = {p:popt[kpmf] for kpmf,p in enumerate(spm_bounds.keys())}
		spm_guess = {p:p0[p] for kpmf,p in enumerate(spm_bounds.keys())}
		return spm_args

	@override
	def get_spm_trace_b(self, b, n):
		trace = Trace()
		for _ in range(max(n, self.n_trace_samples)):
			lcobjb = self.lcobj.get_b(b).copy()
			if len(lcobjb)>0:
				spm_bounds = priors.get_spm_bounds(lcobjb, self.class_names)
				try:
					spm_args = self.get_curvefit_spm_args(lcobjb, spm_bounds, fs.syn_sne_sfunc)
					sne_model = SNeModel(lcobjb, 'spm-mle', spm_bounds, spm_args)
				except ex.CurveFitError:
					sne_model = SNeModel(lcobjb, 'linear', spm_bounds, None)

				trace.append(sne_model)
			else:
				trace.append_null()
		return trace

###################################################################################################################################################

class SynSNeGeneratorMCMC(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
		n_trace_samples=_C.N_TRACE_SAMPLES,
		max_fit_error:float=_C.MAX_FIT_ERROR,
		std_scale:float=_C.OBSE_STD_SCALE,
		min_cadence_days:float=_C.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=_C.MIN_POINTS_LIGHTCURVE_DEFINITION,
		hours_noise_amp:float=_C.HOURS_NOISE_AMP,
		ignored=False,

		mcmc_priors=None,
		n_tune=_C.N_TUNE, # 500, 1000
		n_chains=24,
		thin_by=_C.THIN_BY,
		**kwargs):
		super().__init__(lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
			n_trace_samples,
			max_fit_error,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			hours_noise_amp,
			ignored,
			)
		self.mcmc_priors = mcmc_priors
		self.n_tune = n_tune
		self.n_chains = n_chains
		self.thin_by = thin_by
		
	def get_curvefit_spm_args(self, lcobjb, spm_bounds, _f):
		if len(lcobjb)<_C.MIN_POINTS_LIGHTCURVE_FOR_SPMFIT or lcobjb.get_days_duration()<_C.MIN_DUR_LIGHTCURVE_FOR_SPMFIT:
			raise ex.CurveFitError()
		days, obs, obse = lu.extract_arrays(lcobjb)
		p0 = priors.get_p0(lcobjb, spm_bounds)

		### solve nans
		invalid_indexs = (obs == np.infty) | (obs == -np.infty) | np.isnan(obs)
		obs[invalid_indexs] = 0 # as a patch, use 0
		obse[invalid_indexs] = 1/_C.EPS # as a patch, use a big obs error to null obs

		### bounds
		fit_kwargs = {
			#'method':'lm',
			#'method':'trf',
			#'method':'dogbox',
			#'absolute_sigma':True,
			#'maxfev':1e6,
			'check_finite':True,
			'bounds':([spm_bounds[p][0] for p in spm_bounds.keys()], [spm_bounds[p][-1] for p in spm_bounds.keys()]),
			'ftol':p0['A']/20., # A_guess
			#'ftol':_C.CURVE_FIT_FTOL,
			'sigma':_C.RE_CLOSS_EPS+_C.RE_CLOSS_K*(obse**2),
		}

		### fitting
		try:
			p0_ = [p0[p] for p in spm_bounds.keys()]
			popt, pcov = curve_fit(_f, days, obs, p0=p0_, **fit_kwargs)
		
		except (ValueError, RuntimeError):
			raise ex.CurveFitError()

		spm_args = {spm_p:popt[kpmf] for kpmf,spm_p in enumerate(spm_bounds.keys())}
		spm_guess = {spm_p:p0[spm_p] for kpmf,spm_p in enumerate(spm_bounds.keys())}
		return spm_args

	def get_mcmc_trace(self, lcobjb, spm_bounds, n, _f, b, mle_spm_args):
		days, obs, obse = lu.extract_arrays(lcobjb)
		mcmc_kwargs = {
			'thin_by':self.thin_by,
			'progress':False,
		}
		assert self.n_trace_samples%self.n_chains==0

		theta0 = np.array([[priors.get_spm_gaussian_sphere(mle_spm_args, spm_bounds, k_std=5e-3)[spm_p].rvs() for spm_p in spm_bounds.keys()] for _ in range(self.n_chains)])
		#print(theta0)
		#assert 0
		#theta0 = np.array([[priors.get_spm_random_sphere(mle_spm_args, spm_bounds)[spm_p] for spm_p in spm_bounds.keys()] for _ in range(self.n_chains)])
		#theta0 = np.array([[mle_spm_args[spm_p] for spm_p in spm_bounds.keys()] for _ in range(self.n_chains)])
		#print(theta0.shape)
		
		try:
			another_lcobjs = [self.lcobj.get_b(_b) for _b in self.band_names if _b!=b]
			aux_lcobjb = sum(another_lcobjs)
			aux_spm_bounds = priors.get_spm_bounds(aux_lcobjb, self.class_names)
			aux_spm_args = self.get_curvefit_spm_args(aux_lcobjb, aux_spm_bounds, fs.syn_sne_sfunc)
			#print('aux_spm_args')
			d_theta = [priors.get_spm_gaussian_sphere(aux_spm_args, spm_bounds, k_std=.1)[spm_p] for spm_p in spm_bounds.keys()]

		except ex.CurveFitError:
			raise ex.MCMCError()
			#print('classic d_theta')
			#d_theta = [priors.get_spm_uniform_box(spm_bounds)[spm_p] for spm_p in spm_bounds.keys()]
			#assert 0
			#d_theta = [self.mcmc_priors[b][self.c][spm_p] for spm_p in spm_bounds.keys()]
		
		sampler = emcee.EnsembleSampler(self.n_chains, theta0.shape[-1], fs.log_probability, args=(d_theta, _f, days, obs, obse))
		try:
			sampler.run_mcmc(theta0, (self.n_trace_samples+self.n_tune)//self.n_chains, **mcmc_kwargs)
		except (ValueError, AssertionError, RuntimeError):
			raise ex.MCMCError()

		mcmc_trace = sampler.get_chain(discard=self.n_tune//self.n_chains, flat=True)[::-1] # (n,spms)
		#print('mcmc_trace',mcmc_trace.shape)
		len_mcmc_trace = len(mcmc_trace)
		mcmc_trace = {spm_p:mcmc_trace[:,kp].tolist() for kp,spm_p in enumerate(spm_bounds.keys())}
		return mcmc_trace, len_mcmc_trace

	@override
	def get_spm_trace_b(self, b, n):
		trace = Trace()
		try:
			lcobjb = self.lcobj.get_b(b).copy()
			if len(lcobjb)<=1:
				raise ex.MCMCError()
			spm_bounds = priors.get_spm_bounds(lcobjb, self.class_names)
			try:
				mle_spm_args = self.get_curvefit_spm_args(lcobjb, spm_bounds, fs.syn_sne_sfunc)
			except ex.CurveFitError:
				raise ex.MCMCError()

			mcmc_trace, len_mcmc_trace = self.get_mcmc_trace(lcobjb, spm_bounds, n, fs.syn_sne_sfunc, b, mle_spm_args)
			for k in range(0, len_mcmc_trace):
				spm_args = {p:mcmc_trace[p][k] for p in mcmc_trace.keys()}
				sne_model = SNeModel(lcobjb, 'spm-mcmc', spm_bounds, spm_args)
				trace.append(sne_model)

		except ex.MCMCError:
			for _ in range(max(n, self.n_trace_samples)):
				lcobjb = self.lcobj.get_b(b).copy()
				if len(lcobjb)>0:
					spm_bounds = priors.get_spm_bounds(lcobjb, self.class_names)
					sne_model = SNeModel(lcobjb, 'linear', spm_bounds, None)
					trace.append(sne_model)
				else:
					trace.append_null()

		return trace