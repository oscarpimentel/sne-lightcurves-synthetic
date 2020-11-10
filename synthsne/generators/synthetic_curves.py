from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from scipy.optimize import curve_fit
from . import exceptions as ex
from lchandler.lc_classes import diff_vector
import pymc3 as pm
from . import lc_utils as lu
from . import funcs as f_
from . import bounds as b_
from . import metrics as metrics

###################################################################################################################################################

def get_syn_sne_generator(method_name):
	if method_name=='dummy':
		return SynSNeGenerator
	elif method_name=='curve_fit':
		return SynSNeGeneratorCF
	elif method_name=='mcmc':
		return SynSNeGeneratorMCMC
	else:
		raise Exception(f'no method_name {method_name}')

class Trace():
	def __init__(self, pm_features):
		self.pm_features = pm_features.copy()
		self.reset()

	def reset(self):
		self.pm_args_l = []
		self.pm_bounds_l = []
		self.fit_errors = []
		self.correct_fit_tags = []

	def add(self, pm_args, pm_bounds, correct_fit_tag):
		self.pm_args_l.append(pm_args)
		self.pm_bounds_l.append(pm_bounds)
		self.correct_fit_tags.append(bool(correct_fit_tag))

	def get_fit_errors(self, lcobjb, func):
		for k in range(len(self)):
			fit_error = np.infty
			if self.correct_fit_tags[k]:
				days, obs, obs_error = lu.extract_arrays(lcobjb)
				fit_error = metrics.swmse_error_syn_sne(days, obs, obs_error, func, [self.pm_args_l[k][pmf] for pmf in self.pm_features])

			self.fit_errors.append(fit_error)

	def sort(self):
		assert len(self.fit_errors)==len(self)
		idxs = np.argsort(self.fit_errors).tolist()
		self.pm_args_l = [self.pm_args_l[i] for i in idxs]
		self.pm_bounds_l = [self.pm_bounds_l[i] for i in idxs]
		self.fit_errors = [self.fit_errors[i] for i in idxs]
		self.correct_fit_tags = [self.correct_fit_tags[i] for i in idxs]

	def clip(self, n):
		assert n<=len(self)
		self.pm_args_l = self.pm_args_l[:n]
		self.pm_bounds_l = self.pm_bounds_l[:n]
		self.fit_errors = self.fit_errors[:n]
		self.correct_fit_tags = self.correct_fit_tags[:n]

	def get_valid_errors(self):
		return [self.fit_errors[k] for k in range(len(self)) if self.correct_fit_tags[k]]

	def get_errors_mean(self):
		errors = self.get_valid_errors()
		return None if len(errors)==0 else np.mean(errors)

	def get_errors_std(self):
		errors = self.get_valid_errors()
		return None if len(errors)==0 else np.std(errors)

	def get_error(self, k):
		assert k<len(self)
		return self.fit_errors[k] if self.correct_fit_tags[k] else None

	def __len__(self):
		return len(self.pm_args_l)

	def __getitem__(self, k):
		return self.pm_args_l[k], self.pm_bounds_l[k], self.correct_fit_tags[k]

def override(func): return func
class SynSNeGenerator():
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		uses_new_bounds=True,
		replace_nan_inf:bool=True,
		max_obs_error:float=1e10,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
		):
		#self.pm_features = ['A', 't0', 'gamma', 'f', 'trise', 'tfall']; self.func = f_.syn_sne_func; self.inv_func = f_.inverse_syn_sne_func
		self.pm_features = ['A', 't0', 'gamma', 'f', 'trise', 'tfall', 's']; self.func = f_.syn_sne_sfunc; self.inv_func = f_.inverse_syn_sne_sfunc
		#self.pm_features = ['A', 't0', 'gamma', 'f', 'trise', 'tfall', 'g']; self.func = f_.syn_sne_gfunc; self.inv_func = f_.inverse_syn_sne_gfunc

		self.lcobj = lcobj.copy()
		self.class_names = class_names.copy()
		self.c = self.class_names[lcobj.y]
		self.band_names = band_names.copy()
		self.obse_sampler_bdict = obse_sampler_bdict
		self.length_sampler_bdict = length_sampler_bdict
		
		self.n_trace_samples = n_trace_samples
		self.uses_new_bounds = uses_new_bounds
		self.replace_nan_inf = replace_nan_inf,
		self.max_obs_error = max_obs_error,
		self.std_scale = std_scale
		self.min_cadence_days = min_cadence_days
		self.min_synthetic_len_b = min_synthetic_len_b
		self.min_required_points_to_fit = min_required_points_to_fit
		self.min_obs_bdict = {b:self.obse_sampler_bdict[b].min_obs for b in self.band_names}

	def reset(self):
		pass

	def sample_curves(self, n):
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

		return new_lcobjs, new_smooth_lcojbs, trace_bdict

	def get_pm_trace_b(self, b, n): # override this method
		trace = Trace(self.pm_features)
		for k in range(max(n, self.n_trace_samples)):
			try:
				pm_bounds = b_.get_pm_bounds(self.lcobj.get_b(b), self.class_names, self.uses_new_bounds, self.min_required_points_to_fit)[self.c]
				pm_args = {pmf:np.random.uniform(*pm_bounds[pmf]) for pmf in self.pm_features}
				trace.add(pm_args, pm_bounds, True)
			except ex.TooShortCurveError:
				trace.add(None, None, False)
			
		return trace

	def sample_curves_b(self, b, n):
		lcobjb = self.lcobj.get_b(b)
		trace = self.get_pm_trace_b(b, n)
		trace.get_fit_errors(lcobjb, self.func)
		trace.sort()
		trace.clip(n)
		new_lcobjbs = []
		new_smooth_lcobjbs = []
		curve_sizes = self.length_sampler_bdict[b].sample(n)
		for k in range(n):
			pm_args, pm_bounds, correct_fit_tag = trace[k]
			try:
				if not correct_fit_tag:
					raise ex.TraceError()
				pm_times = b_.get_pm_times(self.func, self.inv_func, lcobjb, pm_args, self.pm_features, pm_bounds, self.min_obs_bdict[b])
				min_obs_threshold = self.min_obs_bdict[b]
				max_obs_threshold = lcobjb.obs.max()*10
				new_lcobjb = self.__sample_curve__(lcobjb, pm_times, pm_args, curve_sizes[k], self.obse_sampler_bdict[b], min_obs_threshold, max_obs_threshold, False)
				new_smooth_lcobjb = self.__sample_curve__(lcobjb, pm_times, pm_args, curve_sizes[k], self.obse_sampler_bdict[b], min_obs_threshold, max_obs_threshold, True)

			except ex.SyntheticCurveTimeoutError:
				trace.correct_fit_tags[k] = False # update
				new_lcobjb = lcobjb.synthetic_copy()
				new_smooth_lcobjb = lcobjb.synthetic_copy()

			except ex.TraceError:
				trace.correct_fit_tags[k] = False # update
				new_lcobjb = lcobjb.synthetic_copy()
				new_smooth_lcobjb = lcobjb.synthetic_copy()

			new_lcobjbs.append(new_lcobjb)
			new_smooth_lcobjbs.append(new_smooth_lcobjb)

		return new_lcobjbs, new_smooth_lcobjbs, trace

	def __sample_curve__(self, lcobjb, pm_times, pm_args, curve_size, obse_sampler, min_obs_threshold, max_obs_threshold,
		uses_smooth_obs:bool=False,
		timeout_counter=1000,
		pm_obs_n=100,
		):
		new_lcobjb = lcobjb.synthetic_copy() # copy
		i = 0
		while True:
			i += 1
			if i>=timeout_counter:
				raise ex.SyntheticCurveTimeoutError()

			### generate times to evaluate
			if uses_smooth_obs:
				new_days = np.linspace(pm_times['ti'], pm_times['tf'], pm_obs_n)
			else:
				### generate days grid according to cadence
				new_day = pm_times['ti']
				new_days = []
				while new_day<pm_times['tf']:
					new_days.append(new_day)
					new_day += self.min_cadence_days
				new_days = np.array(new_days)

				### generate actual observation times
				idxs = np.random.permutation(np.arange(0, len(new_days)))
				new_days = new_days[idxs][:curve_size] # random select
				new_days = np.sort(new_days) # sort

				if len(new_days)<=self.min_synthetic_len_b: # need to be long enough
					continue

			### generate parametric observations
			pm_obs = self.func(new_days, *[pm_args[pmf] for pmf in self.pm_features])
			if pm_obs.min()<min_obs_threshold: # can't have observation above the threshold
				continue

			### resampling obs using obs error
			if uses_smooth_obs:
				new_obse = pm_obs*0
				new_obs = pm_obs
			else:
				new_obse, new_obs = obse_sampler.conditional_sample(pm_obs)
				new_obs = np.clip(np.random.normal(pm_obs, new_obse*self.std_scale), min_obs_threshold, None)
			
			if new_obs.max()>max_obs_threshold: # flux can't be too high
				continue

			new_lcobjb.set_values(new_days, new_obs, new_obse)
			return new_lcobjb

###################################################################################################################################################

class SynSNeGeneratorCF(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		uses_new_bounds=True,
		replace_nan_inf:bool=True,
		max_obs_error:float=1e10,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit

		uses_random_guess:bool=False,
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		cpds_p:float=C_.CPDS_P,
		):
		super().__init__(lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
			n_trace_samples,
			uses_new_bounds,
			replace_nan_inf,
			max_obs_error,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			min_required_points_to_fit,
			)
		self.uses_random_guess = uses_random_guess
		self.hours_noise_amp = hours_noise_amp
		self.cpds_p = cpds_p

	def get_p0(self, lcobjb, pm_bounds):
		days, obs, obs_error = lu.extract_arrays(lcobjb)

		### checks
		if len(days)<C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT: # min points to even try a curve fit
			raise ex.TooShortCurveError()

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
		#A_guess = 1.2*max_flux if not self.uses_random_guess else get_random_mean(pm_bounds['A'][0], pm_bounds['A'][1], frac_r)
		A_guess = 1.2*max_flux if not self.uses_random_guess else get_random_mean(1.2*max_flux, 1.2*max_flux, frac_r)

		### t0
		t0_guess = max_flux_day
		
		### gamma
		mask = obs >= max_flux / 3. #np.percentile(obs, 33)
		gamma_guess = new_days[mask].max() - new_days[mask].min() if mask.sum() > 0 else pm_bounds['gamma'][0]

		### f
		f_guess = 0.5 if not self.uses_random_guess else get_random_mean(pm_bounds['f'][0], pm_bounds['f'][-1], frac_r)
		
		### trise
		trise_guess = (max_flux_day - first_day) / 2.
		
		### tfall
		tfall_guess = 40.

		### s
		s_guess = 1/3.

		### g
		g_guess = 0.5

		### set
		p0 = {
			'A':np.clip(A_guess, pm_bounds['A'][0], pm_bounds['A'][-1]),
			't0':np.clip(t0_guess, pm_bounds['t0'][0], pm_bounds['t0'][-1]),
			'gamma':np.clip(gamma_guess, pm_bounds['gamma'][0], pm_bounds['gamma'][-1]),
			'f':np.clip(gamma_guess, pm_bounds['f'][0], pm_bounds['f'][-1]),
			'trise':np.clip(trise_guess, pm_bounds['trise'][0], pm_bounds['trise'][-1]),
			'tfall':np.clip(tfall_guess, pm_bounds['tfall'][0], pm_bounds['tfall'][-1]),
			's':np.clip(s_guess, pm_bounds['s'][0], pm_bounds['s'][-1]),
			'g':np.clip(g_guess, pm_bounds['g'][0], pm_bounds['g'][-1]),
		}
		return p0

	def get_pm_args(self, lcobjb, pm_bounds):
		days, obs, obs_error = lu.extract_arrays(lcobjb)
		p0 = self.get_p0(lcobjb, pm_bounds)

		### solve nans
		if self.replace_nan_inf:
			invalid_indexs = (obs == np.infty) | (obs == -np.infty) | np.isnan(obs)
			obs[invalid_indexs] = 0 # as a patch, use 0
			obs_error[invalid_indexs] = self.max_obs_error # as a patch, use a big obs error to null obs

		### bounds
		fit_kwargs = {
			#'method':'lm',
			#'method':'trf',
			#'method':'dogbox',
			#'absolute_sigma':True,
			#'maxfev':1e6,
			'check_finite':True,
			'bounds':([pm_bounds[pmf][0] for pmf in self.pm_features], [pm_bounds[pmf][-1] for pmf in self.pm_features]),
			'ftol':p0['A']/20., # A_guess
			'sigma':(obs_error+1e-20),
		}

		### fitting
		try:
			p0_ = [p0[pmf] for pmf in self.pm_features]
			popt, pcov = curve_fit(self.func, days, obs, p0=p0_, **fit_kwargs)
		
		except ValueError:
			raise ex.CurveFitError()

		except RuntimeError:
			raise ex.CurveFitError()

		pm_args = {pmf:popt[kpmf] for kpmf,pmf in enumerate(self.pm_features)}
		pm_guess = {pmf:p0[pmf] for kpmf,pmf in enumerate(self.pm_features)}
		return pm_args
		#return pm_args, pm_guess, pcov, lcobjb

	@override
	def get_pm_trace_b(self, b, n):
		trace = Trace(self.pm_features)
		for _ in range(max(n, self.n_trace_samples)):
			lcobjb = self.lcobj.get_b(b).copy()
			lcobjb.add_day_noise_uniform(self.hours_noise_amp) # add day noise
			lcobjb.add_obs_noise_gaussian(1, self.min_obs_bdict[b]) # add obs noise
			lcobjb.apply_downsampling(self.cpds_p) # curve points downsampling

			try:
				pm_bounds = b_.get_pm_bounds(lcobjb, self.class_names, self.uses_new_bounds, self.min_required_points_to_fit)[self.c]
				pm_args = self.get_pm_args(lcobjb, pm_bounds)
				correct_fit_tag = 1
			except ex.CurveFitError:
				pm_args = None
				correct_fit_tag = 0
			except ex.TooShortCurveError:
				pm_args = None
				correct_fit_tag = 0

			trace.add(pm_args, pm_bounds, correct_fit_tag)
		return trace

###################################################################################################################################################

class SynSNeGeneratorMCMC(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		uses_new_bounds=True,
		replace_nan_inf:bool=True,
		max_obs_error:float=1e10,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit

		cores=2,
		n_tune=500, # 500, 1000
		):
		super().__init__(lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
			n_trace_samples,
			uses_new_bounds,
			replace_nan_inf,
			max_obs_error,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			min_required_points_to_fit,
			)
		self.cores = cores
		self.n_tune = n_tune
		#self.mcmc_trace_bdict = {}
		
	def get_mcmc_trace(self, lcobjb, pm_bounds, n):
		days, obs, obs_error = lu.extract_arrays(lcobjb)
		
		### pymc3
		trace_kwargs = {
			'tune':self.n_tune, # burn-in steps
			'cores':self.cores,
			'progressbar':False,
			'target_accept':1., # 0.95, 1
		}
		import logging; logger = logging.getLogger('pymc3'); logger.setLevel(logging.ERROR) # remove logger
		basic_model = pm.Model()
		with basic_model:
			try:
				A = pm.Uniform('A', *pm_bounds['A'])
				t0 = pm.Uniform('t0', *pm_bounds['t0'])
				#gamma = pm.Normal('gamma', mu=35, sigma=10)
				gamma = pm.Uniform('gamma', *pm_bounds['gamma'])
				#gamma = pm.Normal('gamma', mu=pm_bounds['gamma'][0]+(pm_bounds['gamma'][-1]-pm_bounds['gamma'][0])/2., sigma=10)
				#gamma = pm.Gamma('gamma', alpha=pm_bounds['gamma'][0]+(pm_bounds['gamma'][-1]-pm_bounds['gamma'][0])/2., beta=1)
				f = pm.Uniform('f', 0, 1)
				#f = pm.Beta('f', alpha=2.5, beta=1)
				trise = pm.Uniform('trise', *pm_bounds['trise'])
				tfall = pm.Uniform('tfall', *pm_bounds['tfall'])
				s = pm.Uniform('s', *pm_bounds['s'])
				#g = pm.Bernoulli('g', 0.5)

				pm_obs = pm.Normal('pm_obs', mu=self.func(days, A, t0, gamma, f, trise, tfall, s), sigma=obs_error, observed=obs)
				#pm_obs = pm.Normal('pm_obs', mu=self.func(days, A, t0, gamma, f, trise, tfall), sigma=obs_error, observed=obs)
				#pm_obs = pm.StudentT('pm_obs'. nu=5, mu=pm_obs, sigma=obs_error, observed=obs)

				# trace
				#step = pm.Metropolis()
				#step = pm.NUTS()
				mcmc_trace = pm.sample(max(n, self.n_trace_samples), **trace_kwargs)

			#try:
			#	pass
			except ValueError:
				raise ex.PYMCError()
			except AssertionError:
				raise ex.PYMCError()
			except RuntimeError:
				raise ex.PYMCError()

		return mcmc_trace

	@override
	def get_pm_trace_b(self, b, n):
		trace = Trace(self.pm_features)
		try:
			lcobjb = self.lcobj.get_b(b).copy()
			pm_bounds = b_.get_pm_bounds(lcobjb, self.class_names, self.uses_new_bounds, self.min_required_points_to_fit)[self.c]
			#print('get_mcmc_trace')
			mcmc_trace = self.get_mcmc_trace(lcobjb, pm_bounds, n)
			#print(len(mcmc_trace))
			for k in range(len(mcmc_trace)):
				pm_args = {pmf:mcmc_trace[pmf][-k] for pmf in self.pm_features}
				trace.add(pm_args, pm_bounds, True)

		except ex.PYMCError:
			for _ in range(max(n, self.n_trace_samples)):
				trace.add(None, None, False)
		
		return trace