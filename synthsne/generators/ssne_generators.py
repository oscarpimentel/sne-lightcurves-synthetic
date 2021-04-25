from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import random
import emcee
from scipy.optimize import curve_fit
import scipy.stats as stats
from . import exceptions as ex
from . import lc_utils as lu
from .sne_models import SNeModel
from . import priors as priors
from nested_dict import nested_dict
from .ssne_generator import SynSNeGenerator
from . import functions as fs
from .traces import Trace

###################################################################################################################################################

def override(func): return func # tricky
class SynSNeGeneratorLinear(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
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
				lcobjb.add_day_noise_uniform(self.hours_noise_amp) # add day noise
				lcobjb.add_obs_noise_gaussian(self.min_obs_bdict[b], self.std_scale) # add obs noise

				spm_bounds = priors.get_spm_bounds(lcobjb, self.class_names)
				sne_model = SNeModel(lcobjb, 'linear', spm_bounds, None)
				trace.append(sne_model)
			else:
				trace.append_null()
		return trace

###################################################################################################################################################

class SynSNeGeneratorBSpline(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
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
				lcobjb.add_day_noise_uniform(self.hours_noise_amp) # add day noise
				lcobjb.add_obs_noise_gaussian(self.min_obs_bdict[b], self.std_scale) # add obs noise

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
		n_trace_samples=C_.N_TRACE_SAMPLES,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
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
		if len(lcobjb)<C_.MIN_POINTS_LIGHTCURVE_FOR_SPMFIT or lcobjb.get_days_duration()<C_.MIN_DUR_LIGHTCURVE_FOR_SPMFIT:
			raise ex.CurveFitError()
		days, obs, obse = lu.extract_arrays(lcobjb)
		p0 = priors.get_p0(lcobjb, spm_bounds)

		### solve nans
		invalid_indexs = (obs == np.infty) | (obs == -np.infty) | np.isnan(obs)
		obs[invalid_indexs] = 0 # as a patch, use 0
		obse[invalid_indexs] = 1/C_.EPS # as a patch, use a big obs error to null obs

		### bounds
		fit_kwargs = {
			#'method':'dogbox', # lm trf dogbox
			#'absolute_sigma':True,
			#'maxfev':1e6,
			'check_finite':True,
			'bounds':([spm_bounds[p][0] for p in spm_bounds.keys()], [spm_bounds[p][-1] for p in spm_bounds.keys()]),
			'ftol':p0['A']/20., # A_guess
			#'ftol':C_.CURVE_FIT_FTOL,
			'sigma':obse+C_.EPS,
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
				lcobjb.add_day_noise_uniform(self.hours_noise_amp) # add day noise
				lcobjb.add_obs_noise_gaussian(self.min_obs_bdict[b], self.std_scale) # add obs noise

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
		n_trace_samples=C_.N_TRACE_SAMPLES,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		ignored=False,

		mcmc_priors=None,
		n_tune=C_.N_TUNE, # 500, 1000
		n_chains=24,
		thin_by=C_.THIN_BY,
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
		if len(lcobjb)<C_.MIN_POINTS_LIGHTCURVE_FOR_SPMFIT or lcobjb.get_days_duration()<C_.MIN_DUR_LIGHTCURVE_FOR_SPMFIT:
			raise ex.CurveFitError()
		days, obs, obse = lu.extract_arrays(lcobjb)
		p0 = priors.get_p0(lcobjb, spm_bounds)

		### solve nans
		invalid_indexs = (obs == np.infty) | (obs == -np.infty) | np.isnan(obs)
		obs[invalid_indexs] = 0 # as a patch, use 0
		obse[invalid_indexs] = 1/C_.EPS # as a patch, use a big obs error to null obs

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
			#'ftol':C_.CURVE_FIT_FTOL,
			'sigma':obse+C_.EPS,
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

	def get_mcmc_trace(self, lcobjb, spm_bounds, n, _f, b, mle_spm_args):
		days, obs, obse = lu.extract_arrays(lcobjb)
		mcmc_kwargs = {
			'thin_by':self.thin_by,
			'progress':False,
		}
		assert self.n_trace_samples%self.n_chains==0

		theta0 = np.array([[priors.get_spm_random_sphere(mle_spm_args, spm_bounds)[spm_p] for spm_p in spm_bounds.keys()] for _ in range(self.n_chains)])
		#theta0 = np.array([[mle_spm_args[spm_p] for spm_p in spm_bounds.keys()] for _ in range(self.n_chains)])
		#print(theta0.shape)
		d_theta = [self.mcmc_priors[b][self.c][spm_p] for spm_p in spm_bounds.keys()]
		sampler = emcee.EnsembleSampler(self.n_chains, theta0.shape[-1], fs.log_probability, args=(d_theta, _f, days, obs, obse))
		try:
			sampler.run_mcmc(theta0, (self.n_trace_samples+self.n_tune)//self.n_chains, **mcmc_kwargs)
		except (ValueError, AssertionError, RuntimeError):
			raise ex.MCMCError()

		mcmc_trace = sampler.get_chain(discard=self.n_tune//self.n_chains, flat=True)
		len_mcmc_trace = len(mcmc_trace)
		#print(mcmc_trace.shape)
		mcmc_trace = {p:mcmc_trace[:,kp].tolist() for kp,p in enumerate(spm_bounds.keys())}
		return mcmc_trace, len_mcmc_trace

	@override
	def get_spm_trace_b(self, b, n):
		trace = Trace()
		try:
			lcobjb = self.lcobj.get_b(b).copy()

			### fixme?
			mle_spm_args = nested_dict()
			for _b in self.band_names:
				_lcobjb = self.lcobj.get_b(_b).copy()
				try:
					_spm_bounds = priors.get_spm_bounds(_lcobjb, self.class_names)
					_spm_args = self.get_curvefit_spm_args(_lcobjb, _spm_bounds, fs.syn_sne_sfunc)
					for p in _spm_args.keys():
						mle_spm_args[p][_b] = _spm_args[p]
				except ex.CurveFitError:
					pass

			mle_spm_args = mle_spm_args.to_dict()
			if len(mle_spm_args.keys())==0 or not b in mle_spm_args[list(mle_spm_args.keys())[0]].keys():
				raise ex.MCMCError()

			#print(mle_spm_args)
			for p in mle_spm_args.keys():
				if p in ['trise']:
					mle_spm_args[p] = np.min([mle_spm_args[p][_b] for _b in mle_spm_args[p].keys()])
				#if p in ['t0']:
				#	mle_spm_args[p] = np.max([mle_spm_args[p][_b] for _b in mle_spm_args[p].keys()])
				else:
					mle_spm_args[p] = mle_spm_args[p][b]

			###
			spm_bounds = priors.get_spm_bounds(lcobjb, self.class_names)
			mcmc_trace, len_mcmc_trace = self.get_mcmc_trace(lcobjb, spm_bounds, n, fs.syn_sne_sfunc, b, mle_spm_args)
			for k in range(len_mcmc_trace):
				spm_args = {p:mcmc_trace[p][-k] for p in mcmc_trace.keys()}
				sne_model = SNeModel(lcobjb, 'spm-mcmc', spm_bounds, spm_args)
				trace.append(sne_model)

		except ex.MCMCError:
			for _ in range(max(n, self.n_trace_samples)):
				lcobjb = self.lcobj.get_b(b).copy()
				if len(lcobjb)>0:
					lcobjb.add_day_noise_uniform(self.hours_noise_amp) # add day noise
					lcobjb.add_obs_noise_gaussian(self.min_obs_bdict[b], self.std_scale) # add obs noise

					spm_bounds = priors.get_spm_bounds(lcobjb, self.class_names)
					sne_model = SNeModel(lcobjb, 'linear', spm_bounds, None)
					trace.append(sne_model)
				else:
					trace.append_null()

		return trace