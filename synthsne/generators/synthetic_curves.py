from __future__ import print_function
from __future__ import division
from . import C_

from numba import jit
import numpy as np
import random
import emcee
from scipy.optimize import curve_fit
import scipy.stats as stats
from . import exceptions as ex
from lchandler.lc_classes import diff_vector, get_obs_noise_gaussian
from . import lc_utils as lu
from .sne_models import SNeModel
from . import priors as priors
from flamingchoripan.datascience.xerror import XError
from flamingchoripan.times import Cronometer
from nested_dict import nested_dict

###################################################################################################################################################

def get_random_time_mesh(ti, tf, min_dt):
	if tf<=ti:
		return []
	t0 = ti+np.random.uniform(0, min_dt)
	new_times = []
	while t0<tf:
		new_times.append(t0)
		t0 += min_dt
	return new_times
	
def get_augmented_time_mesh(times, ti, tf, min_dt, extra_times,
	dropout_p=0.0,
	):
	assert dropout_p>=0 and dropout_p<=1

	new_times = [ti-min_dt]+[t for t in np.sort(times) if t>=ti and t<=tf]+[tf+min_dt]
	possible_times = []
	for i in range(0, len(new_times)-1):
		ti_ = new_times[i]
		tf_ = new_times[i+1]
		assert tf_>=ti_
		times_ = get_random_time_mesh(ti_+min_dt, tf_-min_dt, min_dt)
		#print(ti_+min_dt, tf_-min_dt, times_)
		possible_times += times_
	
	possible_times = np.array(possible_times) if extra_times is None else np.random.permutation(possible_times)[:extra_times]
	valid_indexs = np.random.uniform(size=len(possible_times))>=dropout_p
	possible_times = possible_times[valid_indexs]
	augmented_time_mesh = np.sort(np.concatenate([times, possible_times])) # sort
	return augmented_time_mesh

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

###################################################################################################################################################

def get_syn_sne_generator(method_name):
	if method_name=='linear':
		return SynSNeGeneratorLinear
	if method_name=='bspline':
		return SynSNeGeneratorBSpline
	if method_name=='spm-mle':
		return SynSNeGeneratorMLE
	if method_name=='spm-mcmc':
		return SynSNeGeneratorMCMC
	raise Exception(f'no method_name {method_name}')

###################################################################################################################################################

class Trace():
	def __init__(self):
		self.sne_model_l = []
		self.spm_bounds_l = []
		self.fit_errors = []
		self.correct_fit_tags = []

	def add(self, sne_model, spm_bounds, correct_fit_tag):
		self.sne_model_l.append(sne_model)
		self.spm_bounds_l.append(spm_bounds)
		self.correct_fit_tags.append(bool(correct_fit_tag))

	def add_ok(self, sne_model, spm_bounds):
		self.add(sne_model, spm_bounds, True)

	def add_null(self):
		self.add(None, None, False)

	def get_fit_errors(self, lcobjb):
		for k in range(len(self)):
			try:
				if self.correct_fit_tags[k]:
					days, obs, obse = lu.extract_arrays(lcobjb)
					sne_model = self.sne_model_l[k]
					fit_error = sne_model.get_error(days, obs, obse)

			except ex.InterpError:
				self.correct_fit_tags[k] = False

			self.fit_errors.append(fit_error if self.correct_fit_tags[k] else np.infty)

	def sort(self):
		assert len(self.fit_errors)==len(self)
		idxs = np.argsort(self.fit_errors).tolist()
		self.sne_model_l = [self.sne_model_l[i] for i in idxs]
		self.spm_bounds_l = [self.spm_bounds_l[i] for i in idxs]
		self.fit_errors = [self.fit_errors[i] for i in idxs]
		self.correct_fit_tags = [self.correct_fit_tags[i] for i in idxs]

	def clip(self, n):
		assert n<=len(self)
		self.sne_model_l = self.sne_model_l[:n]
		self.spm_bounds_l = self.spm_bounds_l[:n]
		self.fit_errors = self.fit_errors[:n]
		self.correct_fit_tags = self.correct_fit_tags[:n]

	def get_valid_errors(self):
		return [self.fit_errors[k] for k in range(len(self)) if self.correct_fit_tags[k]]

	def get_xerror(self):
		errors = self.get_valid_errors()
		#print(errors)
		return XError(errors)

	def get_xerror_k(self, k):
		assert k>=0 and k<len(self) 
		if self.correct_fit_tags[k] and len(self)>0:
			return XError([self.fit_errors[k]])
		else:
			return XError(None)

	def has_corrects_samples(self):
		return any(self.correct_fit_tags)

	def __len__(self):
		return len(self.sne_model_l)

	def __getitem__(self, k):
		return self.sne_model_l[k], self.spm_bounds_l[k], self.correct_fit_tags[k]

###################################################################################################################################################

def override(func): return func # tricky
class SynSNeGenerator():
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		uses_new_bounds=True,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		ignored=False,
		):
		self.lcobj = lcobj.copy()
		self.class_names = class_names
		self.c = self.class_names[lcobj.y]
		self.band_names = band_names
		self.obse_sampler_bdict = obse_sampler_bdict
		self.uses_estw = uses_estw
		
		self.n_trace_samples = n_trace_samples
		self.uses_new_bounds = uses_new_bounds
		self.max_fit_error = max_fit_error
		self.std_scale = std_scale
		self.min_cadence_days = min_cadence_days
		self.min_synthetic_len_b = min_synthetic_len_b
		self.min_required_points_to_fit = min_required_points_to_fit
		self.hours_noise_amp = hours_noise_amp
		self.ignored = ignored
		self.min_obs_bdict = {b:self.obse_sampler_bdict[b].min_raw_obs for b in self.band_names}

	def reset(self):
		pass

	def sample_curves(self, n):
		cr = Cronometer()
		new_lcobjs = [self.lcobj.copy_only_data() for _ in range(n)] # holders
		new_smooth_lcojbs = [self.lcobj.copy_only_data() for _ in range(n)] # holders
		trace_bdict = {}
		for b in self.band_names:
			new_lcobjbs, new_smooth_lcobjbs, trace = self.sample_curves_b(b, n)
			trace_bdict[b] = trace

			for new_lcobj,new_lcobjb in zip(new_lcobjs, new_lcobjbs):
				new_lcobj.add_sublcobj_b(b, new_lcobjb)

			for new_smooth_lcojb,new_smooth_lcobjb in zip(new_smooth_lcojbs, new_smooth_lcobjbs):
				new_smooth_lcojb.add_sublcobj_b(b, new_smooth_lcobjb)

		return new_lcobjs, new_smooth_lcojbs, trace_bdict, cr.dt_segs()

	@override
	def get_spm_trace_b(self, b, n): # override this method!!!
		trace = Trace()
		for k in range(max(n, self.n_trace_samples)):
			try:
				lcobjb = self.lcobj.get_b(b)
				spm_bounds = priors.get_spm_bounds(lcobjb, self.class_names, self.uses_new_bounds, self.min_required_points_to_fit)
				spm_args = {pmf:np.random.uniform(*spm_bounds[pmf]) for pmf in spm_bounds.keys()}
				trace.add_ok(SNeModel(lcobjb, spm_args), spm_bounds)
			except ex.TooShortCurveError:
				trace.add_null()
			
		return trace

	def sample_curves_b(self, b, n):
		lcobjb = self.lcobj.get_b(b)
		trace = self.get_spm_trace_b(b, n)
		trace.get_fit_errors(lcobjb)
		trace.sort()
		trace.clip(n)
		new_lcobjbs = []
		new_smooth_lcobjbs = []
		#curve_sizes = self.length_sampler_bdict[b].sample(n)
		curve_sizes = [None for k in range(n)]
		for k in range(n):
			sne_model, spm_bounds, correct_fit_tag = trace[k]
			fit_error = trace.fit_errors[k]
			try:
				#print(not correct_fit_tag, self.ignored, fit_error>self.max_fit_error)
				if not correct_fit_tag or self.ignored or fit_error>self.max_fit_error:
					raise ex.TraceError()
				sne_model.get_spm_times(self.min_obs_bdict[b], self.uses_estw)
				min_obs_threshold = self.min_obs_bdict[b]
				new_lcobjb = self._sample_curve(lcobjb, sne_model, curve_sizes[k], self.obse_sampler_bdict[b], min_obs_threshold, False)
				new_smooth_lcobjb = self._sample_curve(lcobjb, sne_model, curve_sizes[k], self.obse_sampler_bdict[b], min_obs_threshold, True)

			except ex.SyntheticCurveTimeoutError:
				trace.correct_fit_tags[k] = False # update
				new_lcobjb = lcobjb.copy()
				new_smooth_lcobjb = lcobjb.copy()

			except ex.InterpError:
				trace.correct_fit_tags[k] = False # update
				new_lcobjb = lcobjb.copy()
				new_smooth_lcobjb = lcobjb.copy()

			except ex.TraceError:
				trace.correct_fit_tags[k] = False # update
				new_lcobjb = lcobjb.copy()
				new_smooth_lcobjb = lcobjb.copy()

			new_lcobjbs.append(new_lcobjb)
			new_smooth_lcobjbs.append(new_smooth_lcobjb)

		return new_lcobjbs, new_smooth_lcobjbs, trace

	def _sample_curve(self, lcobjb, sne_model, curve_size, obse_sampler, min_obs_threshold,
		uses_smooth_obs:bool=False,
		timeout_counter=1000,
		spm_obs_n=100,
		):
		new_lcobjb = lcobjb.synthetic_copy() # copy
		spm_times = sne_model.spm_times
		spm_args = sne_model.spm_args
		i = 0
		while True:
			i += 1
			if i>=timeout_counter:
				raise ex.SyntheticCurveTimeoutError()

			### generate times to evaluate
			if uses_smooth_obs:
				new_days = np.linspace(spm_times['ti'], spm_times['tf'], spm_obs_n)
			else:
				### generate days grid according to cadence
				original_days = lcobjb.days
				#print(spm_times['ti'], spm_times['tf'], original_days)
				#new_days = get_augmented_time_mesh(original_days, spm_times['ti'], spm_times['tf'], self.min_cadence_days, int(len(original_days)*0.5))
				#new_days = get_augmented_time_mesh([], spm_times['ti'], spm_times['tf'], self.min_cadence_days, None, 0.3333)
				new_days = get_augmented_time_mesh([], spm_times['ti'], spm_times['tf'], self.min_cadence_days, int(len(original_days)*1.5))
				
				new_days = new_days+np.random.uniform(-self.hours_noise_amp/24., self.hours_noise_amp/24., len(new_days))
				new_days = np.sort(new_days) # sort

				if len(new_days)<=self.min_synthetic_len_b: # need to be long enough
					continue
					#pass

			### generate parametric observations
			spm_obs = sne_model.evaluate(new_days)
			spm_obs = np.clip(spm_obs, min_obs_threshold, None) # can't have observation above the threshold
			#if spm_obs.min()<min_obs_threshold: # can't have observation above the threshold
				#continue	
			
			### resampling obs using obs error
			if uses_smooth_obs:
				new_obse = np.full(spm_obs.shape, C_.EPS)
				new_obs = spm_obs
			else:
				new_obse, new_obs = obse_sampler.conditional_sample(spm_obs)

				#new_obse = new_obse*0+new_obse[0]# dummy
				#syn_std_scale = 1/10
				syn_std_scale = self.std_scale
				#syn_std_scale = self.std_scale*0.5
				new_obs = get_obs_noise_gaussian(spm_obs, new_obse, min_obs_threshold, syn_std_scale)

			if np.any(np.isnan(new_days)) or np.any(np.isnan(new_obs)) or np.any(np.isnan(new_obse)):
				continue

			new_lcobjb.set_values(new_days, new_obs, new_obse)
			return new_lcobjb

###################################################################################################################################################

class SynSNeGeneratorMLE(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		uses_new_bounds=True,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		ignored=False,

		cpds_p:float=C_.CPDS_P,
		**kwargs):
		super().__init__(lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
			n_trace_samples,
			uses_new_bounds,
			max_fit_error,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			min_required_points_to_fit,
			hours_noise_amp,
			ignored,
			)
		self.cpds_p = cpds_p

	def get_curvefit_spm_args(self, lcobjb, spm_bounds, func):
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
			popt, pcov = curve_fit(func, days, obs, p0=p0_, **fit_kwargs)
		
		except ValueError:
			raise ex.CurveFitError()

		except RuntimeError:
			raise ex.CurveFitError()

		spm_args = {p:popt[kpmf] for kpmf,p in enumerate(spm_bounds.keys())}
		spm_guess = {p:p0[p] for kpmf,p in enumerate(spm_bounds.keys())}
		return spm_args

	@override
	def get_spm_trace_b(self, b, n):
		trace = Trace()
		for _ in range(max(n, self.n_trace_samples)):
			lcobjb = self.lcobj.get_b(b).copy()
			lcobjb.add_day_noise_uniform(self.hours_noise_amp) # add day noise
			lcobjb.add_obs_noise_gaussian(self.min_obs_bdict[b], self.std_scale) # add obs noise
			lcobjb.apply_downsampling(self.cpds_p) # curve points downsampling
			sne_model = SNeModel(lcobjb, None)

			try:
				spm_bounds = priors.get_spm_bounds(lcobjb, self.class_names, self.uses_new_bounds, self.min_required_points_to_fit)
				spm_args = self.get_curvefit_spm_args(lcobjb, spm_bounds, sne_model.func)
				sne_model.spm_args = spm_args.copy()
				trace.add_ok(sne_model, spm_bounds)
			except ex.CurveFitError:
				trace.add_null()
			except ex.TooShortCurveError:
				trace.add_null()
		return trace

###################################################################################################################################################

class SynSNeGeneratorMCMC(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		uses_new_bounds=True,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		ignored=False,

		mcmc_priors=None,
		n_tune=C_.N_TUNE, # 500, 1000
		n_chains=24,
		thin_by=C_.THIN_BY,
		**kwargs):
		super().__init__(lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
			n_trace_samples,
			uses_new_bounds,
			max_fit_error,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			min_required_points_to_fit,
			hours_noise_amp,
			ignored,
			)
		self.mcmc_priors = mcmc_priors
		self.n_tune = n_tune
		self.n_chains = n_chains
		self.thin_by = thin_by
		
	def get_curvefit_spm_args(self, lcobjb, spm_bounds, func):
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
			popt, pcov = curve_fit(func, days, obs, p0=p0_, **fit_kwargs)
		
		except ValueError:
			raise ex.CurveFitError()

		except RuntimeError:
			raise ex.CurveFitError()

		spm_args = {p:popt[kpmf] for kpmf,p in enumerate(spm_bounds.keys())}
		spm_guess = {p:p0[p] for kpmf,p in enumerate(spm_bounds.keys())}
		return spm_args

	def get_mcmc_trace(self, lcobjb, spm_bounds, n, func, b, mle_spm_args):
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
		sampler = emcee.EnsembleSampler(self.n_chains, theta0.shape[-1], log_probability, args=(d_theta, func, days, obs, obse))
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
			sne_model = SNeModel(lcobjb, None) # auxiliar

			mle_spm_args = nested_dict()
			for _b in self.band_names: # fixme?
				_lcobjb = self.lcobj.get_b(_b).copy()
				_spm_bounds = priors.get_spm_bounds(_lcobjb, self.class_names, self.uses_new_bounds, self.min_required_points_to_fit)
				_spm_args = self.get_curvefit_spm_args(_lcobjb, _spm_bounds, sne_model.func)
				for p in sne_model.parameters:
					mle_spm_args[p][_b] = _spm_args[p]

			mle_spm_args = mle_spm_args.to_dict()
			#print(mle_spm_args)
			for p in sne_model.parameters:
				if p in ['trise']:
					mle_spm_args[p] = np.min([mle_spm_args[p][_b] for _b in self.band_names])
				#elif p in ['trise', 'A']:
				#	mle_spm_args[p] = np.min(mle_spm_args[p])
				else:
					mle_spm_args[p] = mle_spm_args[p][b]
					#mle_spm_args[p] = np.mean([mle_spm_args[p][_b] for _b in self.band_names])

			spm_bounds = priors.get_spm_bounds(lcobjb, self.class_names, self.uses_new_bounds, self.min_required_points_to_fit)
			mcmc_trace, len_mcmc_trace = self.get_mcmc_trace(lcobjb, spm_bounds, n, sne_model.func, b, mle_spm_args)
			for k in range(len_mcmc_trace):
				spm_args = {p:mcmc_trace[p][-k] for p in sne_model.parameters}
				trace.add_ok(SNeModel(lcobjb, spm_args), spm_bounds)

		except (ex.MCMCError, ex.TooShortCurveError):
			for _ in range(max(n, self.n_trace_samples)):
				trace.add_null()

		return trace

###################################################################################################################################################

class SynSNeGeneratorLinear(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		uses_new_bounds=True,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		ignored=False,

		cpds_p:float=C_.CPDS_P,
		**kwargs):
		super().__init__(lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
			n_trace_samples,
			uses_new_bounds,
			max_fit_error,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			min_required_points_to_fit,
			hours_noise_amp,
			ignored,
			)
		self.cpds_p = cpds_p

	@override
	def get_spm_trace_b(self, b, n):
		trace = Trace()
		for _ in range(max(n, self.n_trace_samples)):
			lcobjb = self.lcobj.get_b(b).copy()
			lcobjb.add_day_noise_uniform(self.hours_noise_amp) # add day noise
			lcobjb.add_obs_noise_gaussian(self.min_obs_bdict[b], self.std_scale) # add obs noise
			lcobjb.apply_downsampling(self.cpds_p) # curve points downsampling

			try:
				spm_bounds = priors.get_spm_bounds(lcobjb, self.class_names, self.uses_new_bounds, self.min_required_points_to_fit)
				sne_model = SNeModel(lcobjb, None, 'linear')
				trace.add_ok(sne_model, spm_bounds)
			except ex.TooShortCurveError:
				trace.add_null()
		return trace

class SynSNeGeneratorBSpline(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		uses_new_bounds=True,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		ignored=False,

		cpds_p:float=C_.CPDS_P,
		**kwargs):
		super().__init__(lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
			n_trace_samples,
			uses_new_bounds,
			max_fit_error,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			min_required_points_to_fit,
			hours_noise_amp,
			ignored,
			)
		self.cpds_p = cpds_p

	@override
	def get_spm_trace_b(self, b, n):
		trace = Trace()
		for _ in range(max(n, self.n_trace_samples)):
			lcobjb = self.lcobj.get_b(b).copy()
			lcobjb.add_day_noise_uniform(self.hours_noise_amp) # add day noise
			lcobjb.add_obs_noise_gaussian(self.min_obs_bdict[b], self.std_scale) # add obs noise
			lcobjb.apply_downsampling(self.cpds_p) # curve points downsampling

			try:
				spm_bounds = priors.get_spm_bounds(lcobjb, self.class_names, self.uses_new_bounds, self.min_required_points_to_fit)
				sne_model = SNeModel(lcobjb, None, 'bspline')
				trace.add_ok(sne_model, spm_bounds)
			except ex.TooShortCurveError:
				trace.add_null()
		return trace
