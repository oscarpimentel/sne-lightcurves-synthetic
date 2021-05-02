from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from .traces import Trace
from flamingchoripan.times import Cronometer
from . import exceptions as ex
from . import time_meshs as tm
from lchandler.lc_classes import diff_vector, get_obs_noise_gaussian

###################################################################################################################################################

def override(func): return func # tricky
class SynSNeGenerator():
	def __init__(self, lcobj, class_names, band_names, obse_sampler_bdict, uses_estw,
		n_trace_samples=C_.N_TRACE_SAMPLES,
		max_fit_error:float=C_.MAX_FIT_ERROR,
		std_scale:float=C_.OBSE_STD_SCALE,
		min_cadence_days:float=C_.MIN_CADENCE_DAYS,
		min_synthetic_len_b:int=C_.MIN_POINTS_LIGHTCURVE_DEFINITION,
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
		self.max_fit_error = max_fit_error
		self.std_scale = std_scale
		self.min_cadence_days = min_cadence_days
		self.min_synthetic_len_b = min_synthetic_len_b
		self.hours_noise_amp = hours_noise_amp
		self.ignored = ignored
		self.reset()

	def reset(self):
		self.min_obs_bdict = {b:self.obse_sampler_bdict[b].min_raw_obs for b in self.band_names}

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

	def sample_curves_b(self, b, n):
		lcobjb = self.lcobj.get_b(b)
		trace = self.get_spm_trace_b(b, n)
		trace.get_fit_errors(lcobjb)
		#trace.sort()
		trace.clip(n)
		new_lcobjbs = []
		new_smooth_lcobjbs = []
		curve_sizes = [None for k in range(n)]
		for k in range(n):
			sne_model = trace[k]
			fit_error = trace.fit_errors[k]
			try:
				if any([sne_model is None, self.ignored]):
					raise ex.TraceError()
				if fit_error>self.max_fit_error:
					print(f'max_fit_error: {fit_error}')
					raise ex.TraceError()
				sne_model.get_spm_times(self.min_obs_bdict[b], self.uses_estw)
				min_obs_threshold = self.min_obs_bdict[b]
				new_lcobjb = self._sample_curve(lcobjb, sne_model, curve_sizes[k], self.obse_sampler_bdict[b], min_obs_threshold, sne_model.spm_type, False)
				new_smooth_lcobjb = self._sample_curve(lcobjb, sne_model, curve_sizes[k], self.obse_sampler_bdict[b], min_obs_threshold, sne_model.spm_type, True)

			except (ex.SyntheticCurveTimeoutError, ex.TraceError):
				trace.sne_models[k] = None # update
				new_lcobjb = lcobjb.copy()
				new_smooth_lcobjb = lcobjb.copy()

			new_lcobjbs.append(new_lcobjb)
			new_smooth_lcobjbs.append(new_smooth_lcobjb)

		return new_lcobjbs, new_smooth_lcobjbs, trace

	@override
	def get_spm_trace_b(self, b, n): # override this method!!!
		trace = Trace()
		for k in range(max(n, self.n_trace_samples)):
			lcobjb = self.lcobj.get_b(b)
			if len(lcobjb)>0:
				spm_bounds = priors.get_spm_bounds(lcobjb, self.class_names)
				spm_args = {pmf:np.random.uniform(*spm_bounds[pmf]) for pmf in spm_bounds.keys()}
				trace.add_ok(SNeModel(lcobjb, spm_args), spm_bounds)
			else:
				trace.add_null()
			
		return trace

	def _sample_curve(self, lcobjb, sne_model, curve_size, obse_sampler, min_obs_threshold, synthetic_mode,
		uses_smooth_obs:bool=False,
		timeout_counter=10000,
		spm_obs_n=100,
		):
		new_lcobjb = lcobjb.copy() # copy
		new_lcobjb.set_synthetic_mode(synthetic_mode)
		spm_times = sne_model.spm_times
		spm_args = sne_model.spm_args
		i = 0
		while True:
			i += 1
			if i>=timeout_counter:
				#print('SyntheticCurveTimeoutError')
				raise ex.SyntheticCurveTimeoutError()

			### generate times to evaluate
			if uses_smooth_obs:
				new_days = np.linspace(spm_times['ti'], spm_times['tf'], spm_obs_n if len(lcobjb)>1 else 1)
			else:
				### generate days grid according to cadence
				original_days = lcobjb.days
				#print(spm_times['ti'], spm_times['tf'], original_days)
				#new_days = tm.get_augmented_time_mesh(original_days, spm_times['ti'], spm_times['tf'], self.min_cadence_days, int(len(original_days)*0.5))
				#new_days = tm.get_augmented_time_mesh([], spm_times['ti'], spm_times['tf'], self.min_cadence_days, None, 0.3333)
				new_days = tm.get_augmented_time_mesh([], spm_times['ti'], spm_times['tf'], self.min_cadence_days, int(len(original_days)*1.))
				
				new_days = new_days+np.random.uniform(-self.hours_noise_amp/24., self.hours_noise_amp/24., len(new_days))
				new_days = np.sort(new_days) # sort

				if len(new_days)<=self.min_synthetic_len_b: # need to be long enough
					#print('continue1')
					#continue
					pass

			### generate parametric observations
			spm_obs = sne_model.evaluate(new_days)
			if any(spm_obs<=C_.EPS):
				continue
				#spm_obs = np.clip(spm_obs, min_obs_threshold, None) # can't have observation above the threshold
			
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
				#print('continue2')
				continue

			new_lcobjb.set_values(new_days, new_obs, new_obse)
			return new_lcobjb