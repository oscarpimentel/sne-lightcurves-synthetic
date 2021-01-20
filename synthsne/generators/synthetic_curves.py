from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import random
from scipy.optimize import curve_fit
from . import exceptions as ex
from lchandler.lc_classes import diff_vector, get_obs_noise_gaussian
import pymc3 as pm
from . import lc_utils as lu
from .sne_models import SNeModel
from . import bounds as b_
from flamingchoripan.datascience.statistics import XError
from flamingchoripan.times import Cronometer

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
	
def get_augmented_time_mesh(times, ti, tf, min_dt, extra_times):

	new_times = [ti-min_dt]+[t for t in np.sort(times) if t>=ti and t<=tf]+[tf+min_dt]
	possible_times = []
	for i in range(0, len(new_times)-1):
		ti_ = new_times[i]
		tf_ = new_times[i+1]
		assert tf_>=ti_
		times_ = get_random_time_mesh(ti_+min_dt, tf_-min_dt, min_dt)
		#print(ti_+min_dt, tf_-min_dt, times_)
		possible_times += times_
	
	possible_times = np.random.permutation(possible_times)[:extra_times]
	augmented_time_mesh = np.sort(np.concatenate([times, possible_times])) # sort
	return augmented_time_mesh

def get_syn_sne_generator(method_name):
	if method_name=='curvefit':
		return SynSNeGeneratorCF
	if method_name=='mcmc':
		return SynSNeGeneratorMCMC
	if method_name=='tmcmc':
		return SynSNeGeneratorTMCMC
	if method_name=='linear':
		return SynSNeGeneratorLinear
	if method_name=='bspline':
		return SynSNeGeneratorBSpline
	raise Exception(f'no method_name {method_name}')

###################################################################################################################################################

class Trace():
	def __init__(self):
		self.sne_model_l = []
		self.pm_bounds_l = []
		self.fit_errors = []
		self.correct_fit_tags = []

	def add(self, sne_model, pm_bounds, correct_fit_tag):
		self.sne_model_l.append(sne_model)
		self.pm_bounds_l.append(pm_bounds)
		self.correct_fit_tags.append(bool(correct_fit_tag))

	def add_ok(self, sne_model, pm_bounds):
		self.add(sne_model, pm_bounds, True)

	def add_null(self):
		self.add(None, None, False)

	def get_fit_errors(self, lcobjb):
		for k in range(len(self)):
			try:
				if self.correct_fit_tags[k]:
					days, obs, obs_error = lu.extract_arrays(lcobjb)
					sne_model = self.sne_model_l[k]
					fit_error = sne_model.get_error(days, obs, obs_error)

			except ex.InterpError:
				self.correct_fit_tags[k] = False

			self.fit_errors.append(fit_error if self.correct_fit_tags[k] else np.infty)

	def sort(self):
		assert len(self.fit_errors)==len(self)
		idxs = np.argsort(self.fit_errors).tolist()
		self.sne_model_l = [self.sne_model_l[i] for i in idxs]
		self.pm_bounds_l = [self.pm_bounds_l[i] for i in idxs]
		self.fit_errors = [self.fit_errors[i] for i in idxs]
		self.correct_fit_tags = [self.correct_fit_tags[i] for i in idxs]

	def clip(self, n):
		assert n<=len(self)
		self.sne_model_l = self.sne_model_l[:n]
		self.pm_bounds_l = self.pm_bounds_l[:n]
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
		return self.sne_model_l[k], self.pm_bounds_l[k], self.correct_fit_tags[k]

###################################################################################################################################################

def override(func): return func
class SynSNeGenerator():
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		uses_new_bounds=True,
		replace_nan_inf:bool=True,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		ignored=False,
		):
		self.lcobj = lcobj.copy()
		self.class_names = class_names.copy()
		self.c = self.class_names[lcobj.y]
		self.band_names = band_names.copy()
		self.obse_sampler_bdict = obse_sampler_bdict
		self.length_sampler_bdict = length_sampler_bdict
		
		self.n_trace_samples = n_trace_samples
		self.uses_new_bounds = uses_new_bounds
		self.replace_nan_inf = replace_nan_inf,
		self.max_fit_error = max_fit_error
		self.std_scale = std_scale
		self.min_cadence_days = min_cadence_days
		self.min_synthetic_len_b = min_synthetic_len_b
		self.min_required_points_to_fit = min_required_points_to_fit
		self.hours_noise_amp = hours_noise_amp
		self.ignored = ignored
		self.min_obs_bdict = {b:self.obse_sampler_bdict[b].min_obs for b in self.band_names}

	def reset(self):
		pass

	def sample_curves(self, n,
		return_has_corrects_samples=False,
		):
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

		has_corrects_samples = any([trace_bdict[b].has_corrects_samples() for b in self.band_names])
		if return_has_corrects_samples:
			return new_lcobjs, new_smooth_lcojbs, trace_bdict, cr.dt_segs(), has_corrects_samples
		else:
			return new_lcobjs, new_smooth_lcojbs, trace_bdict, cr.dt_segs()

	@override
	def get_pm_trace_b(self, b, n): # override this method
		trace = Trace()
		for k in range(max(n, self.n_trace_samples)):
			try:
				lcobjb = self.lcobj.get_b(b)
				pm_bounds = b_.get_pm_bounds(lcobjb, self.class_names, self.uses_new_bounds, self.min_required_points_to_fit)[self.c]
				pm_args = {pmf:np.random.uniform(*pm_bounds[pmf]) for pmf in pm_bounds.keys()}
				trace.add_ok(SNeModel(lcobjb, pm_args), pm_bounds)
			except ex.TooShortCurveError:
				trace.add_null()
			
		return trace

	def sample_curves_b(self, b, n):
		lcobjb = self.lcobj.get_b(b)
		trace = self.get_pm_trace_b(b, n)
		trace.get_fit_errors(lcobjb)
		trace.sort()
		trace.clip(n)
		new_lcobjbs = []
		new_smooth_lcobjbs = []
		curve_sizes = self.length_sampler_bdict[b].sample(n)
		for k in range(n):
			sne_model, pm_bounds, correct_fit_tag = trace[k]
			fit_error = trace.fit_errors[k]
			try:
				#print(not correct_fit_tag, self.ignored, fit_error>self.max_fit_error)
				if not correct_fit_tag or self.ignored or fit_error>self.max_fit_error:
					raise ex.TraceError()
				sne_model.get_pm_times(self.min_obs_bdict[b])
				min_obs_threshold = self.min_obs_bdict[b]
				new_lcobjb = self.__sample_curve__(lcobjb, sne_model, curve_sizes[k], self.obse_sampler_bdict[b], min_obs_threshold, False)
				new_smooth_lcobjb = self.__sample_curve__(lcobjb, sne_model, curve_sizes[k], self.obse_sampler_bdict[b], min_obs_threshold, True)

			except ex.SyntheticCurveTimeoutError:
				trace.correct_fit_tags[k] = False # update
				new_lcobjb = lcobjb.synthetic_copy()
				new_smooth_lcobjb = lcobjb.synthetic_copy()

			except ex.InterpError:
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

	def __sample_curve__(self, lcobjb, sne_model, curve_size, obse_sampler, min_obs_threshold,
		uses_smooth_obs:bool=False,
		timeout_counter=1000,
		pm_obs_n=100,
		):
		new_lcobjb = lcobjb.synthetic_copy() # copy
		pm_times = sne_model.pm_times
		pm_args = sne_model.pm_args
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
				original_days = lcobjb.days
				days_to_preserve = original_days[:len(original_days)//2]
				#days_to_preserve = np.random.permutation(original_days)[::2]
				#print(pm_times['ti'], pm_times['tf'], original_days)
				#new_days = get_augmented_time_mesh(days_to_preserve, pm_times['ti'], pm_times['tf'], self.min_cadence_days, int(len(original_days)*0.5))
				#new_days = get_augmented_time_mesh(original_days, pm_times['ti'], pm_times['tf'], self.min_cadence_days, int(len(original_days)*0.5))
				new_days = get_augmented_time_mesh(days_to_preserve, pm_times['ti'], pm_times['tf'], self.min_cadence_days, int(len(original_days)*1))
				#new_days = get_augmented_time_mesh([], pm_times['ti'], pm_times['tf'], self.min_cadence_days, int(len(original_days)*1.5))
				new_days = new_days+np.random.uniform(-self.hours_noise_amp/24., self.hours_noise_amp/24., len(new_days))
				new_days = np.sort(new_days) # sort

				if len(new_days)<=self.min_synthetic_len_b: # need to be long enough
					continue
					#pass

			### generate parametric observations
			pm_obs = sne_model.evaluate(new_days)
			if pm_obs.min()<min_obs_threshold: # can't have observation above the threshold
				#continue
				pm_obs = np.clip(pm_obs, min_obs_threshold, None)
			if np.any(np.isnan(pm_obs)):
				print(pm_obs)
				continue
			
			### resampling obs using obs error
			if uses_smooth_obs:
				new_obse = pm_obs*0+C_.EPS
				new_obs = pm_obs
			else:
				new_obse, new_obs = obse_sampler.conditional_sample(pm_obs)
				#new_obse = new_obse*0+new_obse[0]# dummy
				#syn_std_scale = 1/10
				syn_std_scale = self.std_scale
				#syn_std_scale = self.std_scale*0.5
				new_obs = get_obs_noise_gaussian(pm_obs, new_obse, min_obs_threshold, syn_std_scale)

			new_lcobjb.set_values(new_days, new_obs, new_obse)
			return new_lcobjb

###################################################################################################################################################

class SynSNeGeneratorCF(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		uses_new_bounds=True,
		replace_nan_inf:bool=True,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		ignored=False,

		uses_random_guess:bool=False,
		cpds_p:float=C_.CPDS_P,
		):
		super().__init__(lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
			n_trace_samples,
			uses_new_bounds,
			replace_nan_inf,
			max_fit_error,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			min_required_points_to_fit,
			hours_noise_amp,
			ignored,
			)
		self.uses_random_guess = uses_random_guess
		self.cpds_p = cpds_p

	def get_p0(self, lcobjb, pm_bounds):
		days, obs, obs_error = lu.extract_arrays(lcobjb)

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
		f_guess = (pm_bounds['f'][0]+pm_bounds['f'][-1])/2 if not self.uses_random_guess else get_random_mean(pm_bounds['f'][0], pm_bounds['f'][-1], frac_r)
		
		### trise
		trise_guess = (max_flux_day - first_day) / 2.
		
		### tfall
		tfall_guess = 40.

		### s
		#s_guess = 1/3.

		### set
		p0 = {
			'A':np.clip(A_guess, pm_bounds['A'][0], pm_bounds['A'][-1]),
			't0':np.clip(t0_guess, pm_bounds['t0'][0], pm_bounds['t0'][-1]),
			'gamma':np.clip(gamma_guess, pm_bounds['gamma'][0], pm_bounds['gamma'][-1]),
			'f':np.clip(gamma_guess, pm_bounds['f'][0], pm_bounds['f'][-1]),
			'trise':np.clip(trise_guess, pm_bounds['trise'][0], pm_bounds['trise'][-1]),
			'tfall':np.clip(tfall_guess, pm_bounds['tfall'][0], pm_bounds['tfall'][-1]),
			#'s':np.clip(s_guess, pm_bounds['s'][0], pm_bounds['s'][-1]),
		}
		return p0

	def get_pm_args(self, lcobjb, pm_bounds, func):
		days, obs, obs_error = lu.extract_arrays(lcobjb)
		p0 = self.get_p0(lcobjb, pm_bounds)

		### solve nans
		if self.replace_nan_inf:
			invalid_indexs = (obs == np.infty) | (obs == -np.infty) | np.isnan(obs)
			obs[invalid_indexs] = 0 # as a patch, use 0
			obs_error[invalid_indexs] = 1/C_.EPS # as a patch, use a big obs error to null obs

		### bounds
		fit_kwargs = {
			#'method':'lm',
			#'method':'trf',
			#'method':'dogbox',
			#'absolute_sigma':True,
			#'maxfev':1e6,
			'check_finite':True,
			'bounds':([pm_bounds[p][0] for p in pm_bounds.keys()], [pm_bounds[p][-1] for p in pm_bounds.keys()]),
			'ftol':p0['A']/20., # A_guess
			'sigma':obs_error+C_.EPS,
		}

		### fitting
		try:
			p0_ = [p0[p] for p in pm_bounds.keys()]
			popt, pcov = curve_fit(func, days, obs, p0=p0_, **fit_kwargs)
		
		except ValueError:
			raise ex.CurveFitError()

		except RuntimeError:
			raise ex.CurveFitError()

		pm_args = {p:popt[kpmf] for kpmf,p in enumerate(pm_bounds.keys())}
		pm_guess = {p:p0[p] for kpmf,p in enumerate(pm_bounds.keys())}
		return pm_args

	@override
	def get_pm_trace_b(self, b, n):
		trace = Trace()
		for _ in range(max(n, self.n_trace_samples)):
			lcobjb = self.lcobj.get_b(b).copy()
			lcobjb.add_day_noise_uniform(self.hours_noise_amp) # add day noise
			lcobjb.add_obs_noise_gaussian(self.min_obs_bdict[b], self.std_scale) # add obs noise
			lcobjb.apply_downsampling(self.cpds_p) # curve points downsampling
			sne_model = SNeModel(lcobjb, None)

			try:
				pm_bounds = b_.get_pm_bounds(lcobjb, self.class_names, self.uses_new_bounds, self.min_required_points_to_fit)[self.c]
				pm_args = self.get_pm_args(lcobjb, pm_bounds, sne_model.func)
				sne_model.pm_args = pm_args.copy()
				trace.add_ok(sne_model, pm_bounds)
			except ex.CurveFitError:
				trace.add_null()
			except ex.TooShortCurveError:
				trace.add_null()
		return trace

###################################################################################################################################################

class SynSNeGeneratorMCMC(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		uses_new_bounds=True,
		replace_nan_inf:bool=True,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		ignored=False,

		n_tune=1000, # 500, 1000
		mcmc_std_scale=1/2,
		):
		super().__init__(lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
			n_trace_samples,
			uses_new_bounds,
			replace_nan_inf,
			max_fit_error,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			min_required_points_to_fit,
			hours_noise_amp,
			ignored,
			)
		self.n_tune = n_tune
		self.mcmc_std_scale = mcmc_std_scale
		self.obs_dist_name = 'Normal'
		self.obs_dist_kwargs = {}
		
	def get_mcmc_trace(self, lcobjb, pm_bounds, n, func, b):
		days, obs, obs_error = lu.extract_arrays(lcobjb)
		
		### pymc3
		cores = 1
		trace_kwargs = {
			'tune':self.n_tune, # burn-in steps
			'cores':cores,
			'chains':cores,
			#'pickle_backend':'dill',
			'progressbar':False,
			'target_accept':1., # 0.95, 1
		}
		import logging; logger = logging.getLogger('pymc3'); logger.setLevel(logging.ERROR) # remove logger
		basic_model = pm.Model()
		with basic_model:
			#t0 = pm.Uniform('t0', *pm_bounds['t0'])
			#A = pm.Uniform('A', *pm_bounds['A'])
			#gamma = pm.Uniform('gamma', *pm_bounds['gamma'])
			f = pm.Uniform('f', *pm_bounds['f'])
			#trise = pm.Uniform('trise', *pm_bounds['trise'])
			#tfall = pm.Uniform('tfall', *pm_bounds['tfall'])

			if self.c in ['SNIa']:
				if b=='g':
					t0 = pm.Normal('t0', mu=4.659249210315087, sigma=5.350418860022723) # \normdist{4.66}{5.35}
					A = pm.Gamma('A', alpha=1.562125821812763, beta=3.6040387045799616)+0 # \gammadist{1.56}{3.60}
					gamma = pm.Gamma('gamma', alpha=3.7586176301789584, beta=0.1526237797283781)+1 # \gammadist{3.76}{0.15}
					trise = pm.Gamma('trise', alpha=2.9470430925199564, beta=1.101958662096775)+1 # \gammadist{2.95}{1.10}
					tfall = pm.Gamma('tfall', alpha=1.6528941805791997, beta=0.03350165441283507)+1 # \gammadist{1.65}{0.03}
				if b=='r':
					t0 = pm.Normal('t0', mu=5.541947751251396, sigma=6.524875027743679) # \normdist{5.54}{6.52}
					A = pm.Gamma('A', alpha=1.6790249451644361, beta=3.8936870678729356)+0 # \gammadist{1.68}{3.89}
					gamma = pm.Gamma('gamma', alpha=1.6108986496450433, beta=0.06282197060009514)+1 # \gammadist{1.61}{0.06}
					trise = pm.Gamma('trise', alpha=2.374464471671413, beta=0.7713128797336563)+1 # \gammadist{2.37}{0.77}
					tfall = pm.Gamma('tfall', alpha=1.9620491711509604, beta=0.045150231556847276)+1 # \gammadist{1.96}{0.05}
			if self.c in ['allSNII']:
				if b=='g':
					t0 = pm.Normal('t0', mu=1.759587311936204, sigma=13.614984487593611) # \normdist{1.76}{13.61}
					A = pm.Gamma('A', alpha=1.2515158668340511, beta=3.21828026368239)+0 # \gammadist{1.25}{3.22}
					gamma = pm.Gamma('gamma', alpha=1.795487203263114, beta=0.041871088859673965)+1 # \gammadist{1.80}{0.04}
					trise = pm.Gamma('trise', alpha=0.6841141896892713, beta=0.09992353407967454)+1 # \gammadist{0.68}{0.10}
					tfall = pm.Gamma('tfall', alpha=1.6909429790746184, beta=0.030268964809497756)+1 # \gammadist{1.69}{0.03}
				if b=='r':
					t0 = pm.Normal('t0', mu=5.211535387638031, sigma=21.756321053109396) # \normdist{5.21}{21.76}
					A = pm.Gamma('A', alpha=1.3495104688378858, beta=3.2695897856185576)+0 # \gammadist{1.35}{3.27}
					gamma = pm.Gamma('gamma', alpha=2.540175116166474, beta=0.03700858201070383)+1 # \gammadist{2.54}{0.04}
					trise = pm.Gamma('trise', alpha=0.6082321491931155, beta=0.05403599438807471)+1 # \gammadist{0.61}{0.05}
					tfall = pm.Gamma('tfall', alpha=1.2128423538558193, beta=0.024006366171282486)+1 # \gammadist{1.21}{0.02}
			if self.c in ['SNIbc']:
				if b=='g':
					t0 = pm.Normal('t0', mu=5.033365864401862, sigma=7.669319215573517) # \normdist{5.03}{7.67}
					A = pm.Gamma('A', alpha=1.4109594281817543, beta=3.74937921017218)+0 # \gammadist{1.41}{3.75}
					gamma = pm.Gamma('gamma', alpha=2.2987206277006735, beta=0.10032329779991193)+1 # \gammadist{2.30}{0.10}
					trise = pm.Gamma('trise', alpha=1.1032160625121028, beta=0.29058543430646216)+1 # \gammadist{1.10}{0.29}
					tfall = pm.Gamma('tfall', alpha=1.9669165705074316, beta=0.034451473899712134)+1 # \gammadist{1.97}{0.03}
				if b=='r':
					t0 = pm.Normal('t0', mu=4.122916374468605, sigma=7.267432859863303) # \normdist{4.12}{7.27}
					A = pm.Gamma('A', alpha=1.5053383016524007, beta=3.1682813652980086)+0 # \gammadist{1.51}{3.17}
					gamma = pm.Gamma('gamma', alpha=3.784271901221374, beta=0.12049469461793495)+1 # \gammadist{3.78}{0.12}
					trise = pm.Gamma('trise', alpha=1.4355863028350293, beta=0.38867555527313746)+1 # \gammadist{1.44}{0.39}
					tfall = pm.Gamma('tfall', alpha=2.1960393518180124, beta=0.04112048163743286)+1 # \gammadist{2.20}{0.04}
			if self.c in ['SLSN']:
				if b=='g':
					t0 = pm.Normal('t0', mu=12.254579252483346, sigma=10.957857182559481) # \normdist{12.25}{10.96}
					A = pm.Gamma('A', alpha=1.7135351716213902, beta=6.472298399471126)+0 # \gammadist{1.71}{6.47}
					gamma = pm.Gamma('gamma', alpha=7.023971454464478, beta=0.08791106305890829)+1 # \gammadist{7.02}{0.09}
					trise = pm.Gamma('trise', alpha=2.1078879847572223, beta=0.16823310043034334)+1 # \gammadist{2.11}{0.17}
					tfall = pm.Gamma('tfall', alpha=1.516834126009786, beta=0.021816523764273135)+1 # \gammadist{1.52}{0.02}
				if b=='r':
					t0 = pm.Normal('t0', mu=16.22933107923224, sigma=11.28508369834741) # \normdist{16.23}{11.29}
					A = pm.Gamma('A', alpha=2.2299490311194714, beta=8.94989782394901)+0 # \gammadist{2.23}{8.95}
					gamma = pm.Gamma('gamma', alpha=4.95275281713852, beta=0.06207661845103551)+1 # \gammadist{4.95}{0.06}
					trise = pm.Gamma('trise', alpha=2.994833874789099, beta=0.244820659028758)+1 # \gammadist{2.99}{0.24}
					tfall = pm.Gamma('tfall', alpha=1.3136037028978675, beta=0.024218584751623296)+1 # \gammadist{1.31}{0.02}

			pm_obs = getattr(pm, self.obs_dist_name)('pm_obs', mu=func(days, A, t0, gamma, f, trise, tfall), sigma=obs_error*self.mcmc_std_scale, observed=obs, **self.obs_dist_kwargs)

			try:
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
			except RuntimeError: # Chain failed.
				raise ex.PYMCError()

		return mcmc_trace

	@override
	def get_pm_trace_b(self, b, n):
		trace = Trace()
		try:
			lcobjb = self.lcobj.get_b(b).copy()
			sne_model = SNeModel(lcobjb, None) # auxiliar
			pm_bounds = b_.get_pm_bounds(lcobjb, self.class_names, self.uses_new_bounds, self.min_required_points_to_fit)[self.c]
			#print('get_mcmc_trace')
			mcmc_trace = self.get_mcmc_trace(lcobjb, pm_bounds, n, sne_model.func, b)
			#print(len(mcmc_trace))
			for k in range(len(mcmc_trace)):
				pm_args = {p:mcmc_trace[p][-k] for p in sne_model.parameters}
				trace.add_ok(SNeModel(lcobjb, pm_args), pm_bounds)

		except ex.PYMCError:
			for _ in range(max(n, self.n_trace_samples)):
				trace.add_null()
		
		except ex.TooShortCurveError:
			for _ in range(max(n, self.n_trace_samples)):
				trace.add_null()

		return trace

###################################################################################################################################################

class SynSNeGeneratorTMCMC(SynSNeGeneratorMCMC):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		uses_new_bounds=True,
		replace_nan_inf:bool=True,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		ignored=False,

		n_tune=1000, # 500, 1000
		mcmc_std_scale=1/2,
		):
		super().__init__(lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
			n_trace_samples,
			uses_new_bounds,
			replace_nan_inf,
			max_fit_error,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			min_required_points_to_fit,
			hours_noise_amp,
			ignored,

			n_tune,
			mcmc_std_scale,
			)
		self.obs_dist_name = 'StudentT'
		self.obs_dist_kwargs = {
			'nu':5,
		}

###################################################################################################################################################

class SynSNeGeneratorLinear(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		uses_new_bounds=True,
		replace_nan_inf:bool=True,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		ignored=False,

		cpds_p:float=C_.CPDS_P,
		):
		super().__init__(lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
			n_trace_samples,
			uses_new_bounds,
			replace_nan_inf,
			1/C_.EPS,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			min_required_points_to_fit,
			hours_noise_amp,
			ignored,
			)
		self.cpds_p = cpds_p

	@override
	def get_pm_trace_b(self, b, n):
		trace = Trace()
		for _ in range(max(n, self.n_trace_samples)):
			lcobjb = self.lcobj.get_b(b).copy()
			lcobjb.add_day_noise_uniform(self.hours_noise_amp) # add day noise
			lcobjb.add_obs_noise_gaussian(self.min_obs_bdict[b], self.std_scale) # add obs noise
			lcobjb.apply_downsampling(self.cpds_p) # curve points downsampling

			try:
				pm_bounds = b_.get_pm_bounds(lcobjb, self.class_names, self.uses_new_bounds, self.min_required_points_to_fit)[self.c]
				sne_model = SNeModel(lcobjb, None, 'linear')
				trace.add_ok(sne_model, pm_bounds)
			except ex.TooShortCurveError:
				trace.add_null()
		return trace

class SynSNeGeneratorBSpline(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		uses_new_bounds=True,
		replace_nan_inf:bool=True,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
		ignored=False,

		cpds_p:float=C_.CPDS_P,
		):
		super().__init__(lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
			n_trace_samples,
			uses_new_bounds,
			replace_nan_inf,
			1/C_.EPS,
			std_scale,
			min_cadence_days,
			min_synthetic_len_b,
			min_required_points_to_fit,
			hours_noise_amp,
			ignored,
			)
		self.cpds_p = cpds_p

	@override
	def get_pm_trace_b(self, b, n):
		trace = Trace()
		for _ in range(max(n, self.n_trace_samples)):
			lcobjb = self.lcobj.get_b(b).copy()
			lcobjb.add_day_noise_uniform(self.hours_noise_amp) # add day noise
			lcobjb.add_obs_noise_gaussian(self.min_obs_bdict[b], self.std_scale) # add obs noise
			lcobjb.apply_downsampling(self.cpds_p) # curve points downsampling

			try:
				pm_bounds = b_.get_pm_bounds(lcobjb, self.class_names, self.uses_new_bounds, self.min_required_points_to_fit)[self.c]
				sne_model = SNeModel(lcobjb, None, 'bspline')
				trace.add_ok(sne_model, pm_bounds)
			except ex.TooShortCurveError:
				trace.add_null()
		return trace
