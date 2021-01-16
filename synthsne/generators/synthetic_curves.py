from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
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

def get_syn_sne_generator(method_name):
	if method_name=='uniformprior':
		return SynSNeGenerator
	if method_name=='curvefit':
		return SynSNeGeneratorCF
	if method_name=='mcmc':
		return SynSNeGeneratorMCMC
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
		max_obs_error:float=1e10,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,
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
		self.max_obs_error = max_obs_error,
		self.std_scale = std_scale
		self.min_cadence_days = min_cadence_days
		self.min_synthetic_len_b = min_synthetic_len_b
		self.min_required_points_to_fit = min_required_points_to_fit
		self.hours_noise_amp = hours_noise_amp
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
			try:
				if not correct_fit_tag:
					raise ex.TraceError()
				sne_model.get_pm_times(self.min_obs_bdict[b])
				min_obs_threshold = self.min_obs_bdict[b]
				max_obs_threshold = lcobjb.obs.max()*10
				new_lcobjb = self.__sample_curve__(lcobjb, sne_model, curve_sizes[k], self.obse_sampler_bdict[b], min_obs_threshold, max_obs_threshold, False)
				new_smooth_lcobjb = self.__sample_curve__(lcobjb, sne_model, curve_sizes[k], self.obse_sampler_bdict[b], min_obs_threshold, max_obs_threshold, True)

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

	def __sample_curve__(self, lcobjb, sne_model, curve_size, obse_sampler, min_obs_threshold, max_obs_threshold,
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
				new_day = pm_times['ti']
				new_days = []
				while new_day<pm_times['tf']:
					new_days.append(new_day)
					new_day += self.min_cadence_days
				new_days = np.array(new_days)

				### generate actual observation times
				idxs = np.random.permutation(np.arange(0, len(new_days)))
				new_days = new_days[idxs][:curve_size] # random select
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

			### resampling obs using obs error
			if uses_smooth_obs:
				new_obse = pm_obs*0
				new_obs = pm_obs
			else:
				new_obse, new_obs = obse_sampler.conditional_sample(pm_obs)
				new_obs = get_obs_noise_gaussian(pm_obs, new_obse, min_obs_threshold)
			
			if new_obs.max()>max_obs_threshold: # flux can't be too high
				#continue
				new_obs = np.clip(new_obs, None, max_obs_threshold)

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
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,

		uses_random_guess:bool=False,
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
			hours_noise_amp,
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
			obs_error[invalid_indexs] = self.max_obs_error # as a patch, use a big obs error to null obs

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
		max_obs_error:float=1e10,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
		hours_noise_amp:float=C_.HOURS_NOISE_AMP,

		n_tune=1000, # 500, 1000
		mcmc_std_scale=1/2,
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
			hours_noise_amp,
			)
		self.n_tune = n_tune
		self.mcmc_std_scale = mcmc_std_scale
		
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

			if b=='g':
				if self.c in ['SNIa']:
					t0 = pm.Normal('t0', mu=4.825709378956986, sigma=5.5819258725914205) # SNIa-g
					A = pm.Gamma('A', alpha=1.6416077429529141, beta=3.6473248796600015) # SNIa-g
					gamma = pm.Gamma('gamma', alpha=5.670962797825013, beta=0.24024905611249736) # SNIa-g
					#f = pm.Beta('f', alpha=4.29965582517183, beta=1.9284849192939126) # SNIa-g
					trise = pm.Gamma('trise', alpha=5.595808874663219, beta=1.5279304959702806) # SNIa-g
					tfall = pm.Gamma('tfall', alpha=1.944543716900337, beta=0.058638009823029566) # SNIa-g

				elif self.c in ['allSNII']:
					t0 = pm.Normal('t0', mu=0.5196854721845471, sigma=14.954644767893033) # allSNII-g
					A = pm.Gamma('A', alpha=1.3606902363613893, beta=3.1506181917202007) # allSNII-g
					gamma = pm.Gamma('gamma', alpha=2.0575191959006256, beta=0.049067299497742516) # allSNII-g
					#f = pm.Beta('f', alpha=5.517567658613803, beta=3.4192216528164985) # allSNII-g
					trise = pm.Gamma('trise', alpha=1.19885811869537, beta=0.18673896508479088) # allSNII-g
					tfall = pm.Gamma('tfall', alpha=2.155839081652375, beta=0.039329800220804276) # allSNII-g

				elif self.c in ['SNIbc']:
					t0 = pm.Normal('t0', mu=3.3384858933217845, sigma=7.624493623970253) # SNIbc-g
					A = pm.Gamma('A', alpha=1.575102705300546, beta=3.355682608500827) # SNIbc-g
					gamma = pm.Gamma('gamma', alpha=3.342107477729543, beta=0.1352085804247672) # SNIbc-g
					#f = pm.Beta('f', alpha=10.33949951351907, beta=4.151503663373921) # SNIbc-g
					trise = pm.Gamma('trise', alpha=2.008292364792166, beta=0.4563938765178107) # SNIbc-g
					tfall = pm.Gamma('tfall', alpha=2.0627599185205687, beta=0.04397957938789728) # SNIbc-g

				elif self.c in ['SLSN']:
					t0 = pm.Normal('t0', mu=12.588163077473537, sigma=12.066773055811744) # SLSN-g
					A = pm.Gamma('A', alpha=1.771829082671391, beta=6.621546480698072) # SLSN-g
					gamma = pm.Gamma('gamma', alpha=5.625900294956408, beta=0.07578323851762474) # SLSN-g
					#f = pm.Beta('f', alpha=2.4423614173790233, beta=1.6441707267237269) # SLSN-g
					trise = pm.Gamma('trise', alpha=2.4499984639440893, beta=0.18187128125710836) # SLSN-g
					tfall = pm.Gamma('tfall', alpha=1.8844139580199561, beta=0.025067680598546395) # SLSN-g

			elif b=='r':
				if self.c in ['SNIa']:
					t0 = pm.Normal('t0', mu=5.379485383389309, sigma=7.29032215437369) # SNIa-r
					A = pm.Gamma('A', alpha=1.6778354736955787, beta=3.7088123480935087) # SNIa-r
					gamma = pm.Gamma('gamma', alpha=2.354567627457444, beta=0.09860737507829256) # SNIa-r
					#f = pm.Beta('f', alpha=4.6172014795049625, beta=3.274256204806697) # SNIa-r
					trise = pm.Gamma('trise', alpha=4.6870544298725925, beta=1.1467907919215876) # SNIa-r
					tfall = pm.Gamma('tfall', alpha=2.4612354161443593, beta=0.054316713558019036) # SNIa-r

				elif self.c in ['allSNII']:
					t0 = pm.Normal('t0', mu=3.1387199838729063, sigma=19.41880423270187) # allSNII-r
					A = pm.Gamma('A', alpha=1.2888761480471171, beta=3.080107280645589) # allSNII-r
					gamma = pm.Gamma('gamma', alpha=2.7240962002355666, beta=0.04006039829037667) # allSNII-r
					#f = pm.Beta('f', alpha=1.8060630592662028, beta=1.5000740145987317) # allSNII-r
					trise = pm.Gamma('trise', alpha=0.957971037924409, beta=0.10691699663414807) # allSNII-r
					tfall = pm.Gamma('tfall', alpha=1.6057264458471743, beta=0.0324349410682724) # allSNII-r

				elif self.c in ['SNIbc']:
					t0 = pm.Normal('t0', mu=3.6917566111434446, sigma=8.786486396379752) # SNIbc-r
					A = pm.Gamma('A', alpha=1.5254395920542172, beta=3.101119856082535) # SNIbc-r
					gamma = pm.Gamma('gamma', alpha=3.7424429154205496, beta=0.12022264367419071) # SNIbc-r
					#f = pm.Beta('f', alpha=7.984311178323888, beta=4.034154880176986) # SNIbc-r
					trise = pm.Gamma('trise', alpha=3.406856606765631, beta=0.7401374376294634) # SNIbc-r
					tfall = pm.Gamma('tfall', alpha=2.2885477616588203, beta=0.04587396908183451) # SNIbc-r

				elif self.c in ['SLSN']:
					t0 = pm.Normal('t0', mu=19.47625113642454, sigma=15.171995561664644) # SLSN-r
					A = pm.Gamma('A', alpha=2.788407879494774, beta=9.977101267853243) # SLSN-r
					gamma = pm.Gamma('gamma', alpha=5.263302368183219, beta=0.06559384688320556) # SLSN-r
					#f = pm.Beta('f', alpha=2.9329086411857928, beta=2.1757995973727344) # SLSN-r
					trise = pm.Gamma('trise', alpha=3.5177594498008986, beta=0.2477455847152269) # SLSN-r
					tfall = pm.Gamma('tfall', alpha=2.0631105735232627, beta=0.029520368725583696) # SLSN-r

			#pm_obs = pm.Normal('pm_obs', mu=func(days, A, t0, gamma, f, trise, tfall), sigma=obs_error*self.mcmc_std_scale, observed=obs)
			pm_obs = pm.StudentT('pm_obs', nu=5, mu=func(days, A, t0, gamma, f, trise, tfall), sigma=obs_error*self.mcmc_std_scale, observed=obs)

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

class SynSNeGeneratorLinear(SynSNeGenerator):
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, length_sampler_bdict,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		uses_new_bounds=True,
		replace_nan_inf:bool=True,
		max_obs_error:float=1e10,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
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
			hours_noise_amp,
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
		max_obs_error:float=1e10,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
		min_required_points_to_fit:int=C_.MIN_POINTS_LIGHTCURVE_TO_PMFIT, # min points to even try a curve fit
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
			hours_noise_amp,
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
