from __future__ import print_function
from __future__ import division
from . import _C

import math
import numpy as np
import scipy
import scipy.stats as stats
from sklearn import preprocessing as prep
from fuzzytools.datascience.statistics import dropout_extreme_percentiles, get_linspace_ranks
from sklearn.decomposition import PCA

###################################################################################################################################################

class CustomRotor():
	def __init__(self, m, n):
		self.m = m
		assert math.atan(m)>0 and math.atan(m)<np.pi/2
		self.a = np.pi/2-math.atan(m)
		self.n = n
		self.rot = np.linalg.inv(np.array([[np.cos(self.a), np.sin(self.a)], [np.cos(self.a+np.pi/2), np.sin(self.a+np.pi/2)]]))
		self.inv_rot = np.linalg.inv(self.rot)

	def transform(self, obse, obs):
		x = np.concatenate([obse[:,None], obs[:,None]], axis=-1)
		assert x.shape[-1]==2
		x[:,1] = x[:,1]-self.n
		x = (self.rot@x.T).T
		x[:,0] = -1*x[:,0]
		return x[:,0], x[:,1]

	def inverse_transform(self, obse, obs):
		x = np.concatenate([obse[:,None], obs[:,None]], axis=-1)
		assert x.shape[-1]==2
		x[:,0] = -1*x[:,0]
		x = (self.inv_rot@x.T).T
		x[:,1] = x[:,1]+self.n
		return x[:,0], x[:,1]

###################################################################################################################################################

class ObsErrorConditionalSampler():
	def __init__(self, lcset, b:str,
		samples_per_range:int=50,
		rank_threshold=0.04,
		dist_threshold=5e-4,
		neighborhood_n=10,
		):
		self.b = b
		self.samples_per_range = samples_per_range
		self.rank_threshold = rank_threshold
		self.dist_threshold = dist_threshold
		self.neighborhood_n = neighborhood_n
		self.raw_obs = np.concatenate([lcobj.get_b(b).obs for lcobj in lcset.get_lcobjs()])
		self.raw_obse = np.concatenate([lcobj.get_b(b).obse for lcobj in lcset.get_lcobjs()])
		self.min_raw_obs = self.raw_obs.min()
		self.max_raw_obs = self.raw_obs.max()
		self.min_raw_obse = self.raw_obse.min()
		self.max_raw_obse = self.raw_obse.max()
		assert self.min_raw_obs>0
		assert self.min_raw_obse>0
		self.reset()
	
	def reset(self):
		### fit diagonal line
		self.get_m_n()

		### rotate space & clip
		p = 1
		self.obse, self.obs = self.rotor.transform(self.raw_obse, self.raw_obs)
		valid_indexs = np.where(
			#(self.obse>=0) &
			#(self.obs>np.percentile(self.obs, p)) &
			(self.obs<np.percentile(self.obs, 100-p))
		)
		self.obs = self.obs[valid_indexs]
		self.obse = self.obse[valid_indexs]
		self.min_obs = self.obs.min()
		self.max_obs = self.obs.max()
		self.min_obse = self.obse.min()
		self.max_obse = self.obse.max()

		### generate obs grid
		self.rank_ranges, self.obs_indexs_per_range, self.ranks = get_linspace_ranks(self.obs, self.samples_per_range)
		self.distrs = [self.get_fitted_distr(obs_indexs, k) for k,obs_indexs in enumerate(self.obs_indexs_per_range)]

	def clean(self):
		self.raw_obse = None
		self.raw_obs = None
		self.obse = None
		self.obs = None
		return self
	
	def get_m_n(self):
		if 0:
			obse = []
			obs = []
			for k1 in range(len(self.raw_obse)):
				obse1 = self.raw_obse[k1]
				obs1 = self.raw_obs[k1]
				if obs1>self.rank_threshold:
					continue
				neighborhood = 0
				for k2 in range(len(self.raw_obse)):
					if k1==k2:
						continue
					obse2 = self.raw_obse[k2]
					obs2 = self.raw_obs[k2]
					dist = np.linalg.norm(np.array([obse1-obse2, obs1-obs2]))
					if dist<=self.dist_threshold:
						neighborhood += 1
						if neighborhood>=self.neighborhood_n:
							obse.append(obse1)
							obs.append(obs1)
							break

			assert len(obse)>0
			obse = np.array(obse)
			obs = np.array(obs)
			rank_ranges, index_per_range, ranks = get_linspace_ranks(obs, self.samples_per_range)
		else:
			obse = np.array(self.raw_obse)
			obs = np.array(self.raw_obs)
			rank_ranges, index_per_range, ranks = get_linspace_ranks(obs, self.samples_per_range)

		self.lr_x = []
		self.lr_y = []
		for k,indexs in enumerate(index_per_range):
			if len(indexs[0])==0:
				continue
			sub_obse = obse[indexs]
			sub_obs = obs[indexs]
			#self.lr_x.append(np.argmax(sub_obse))
			self.lr_x.append(np.percentile(sub_obse, 50))
			self.lr_y.append(np.percentile(sub_obs, 50))

		#print(self.lr_x, self.lr_y)
		#slope, intercept, r_value, p_value, std_err = stats.linregress(self.raw_obse, self.raw_obs)
		
		x0 = np.percentile(self.raw_obse, 5)
		y0 = np.percentile(self.raw_obs, 95)
		valid_indexs = self.raw_obs<y0
		self.lr_x = self.raw_obse[valid_indexs]#[self.raw_obse>x0]
		self.lr_y = self.raw_obs[valid_indexs]#[self.raw_obs>y0]
		mode = 'pca'
		if mode=='pca':
			pca_x = np.concatenate([self.lr_x[...,None], self.lr_y[...,None]], axis=-1)
			#print(pca_x.shape)
			pca = PCA(n_components=1)
			pca.fit(pca_x)
			slope = pca.components_[0][1]/pca.components_[0][0]
			intercept = 0-pca.mean_[1]
			#print(a.shape, b.shape)
			#assert 0
		else:
			slope, intercept, r_value, p_value, std_err = stats.linregress(self.lr_x, self.lr_y)
		
		#print(slope)
		self.m = slope
		self.n = intercept
		self.rotor = CustomRotor(self.m, self.n)
		
	def get_fitted_distr(self, obs_indexs, k):
		### clean by percentile
		obse_values = self.obse[obs_indexs]
		p = (1-np.exp(-k*1))*1
		#p = 5
		obse_values,_ = dropout_extreme_percentiles(obse_values, p, mode='both')
		distr = getattr(stats, 'norm') # gamma, norm, skewnorm
		params = distr.fit(obse_values)
		#params = distr.fit(obse_values, floc=0)
		return {'distr':distr, 'params':params}
	
	def get_percentile_range(self, obs):
		return np.where(np.clip(obs, None, self.max_obs)<=self.rank_ranges[:,1])[0][0]
		
	def conditional_sample_i(self, obs):
		new_obse = np.array([0])
		new_obs = np.array([obs])
		_,new_obs = self.rotor.transform(new_obse, new_obs)
		d = self.distrs[self.get_percentile_range(new_obs)]
		new_obse = d['distr'].rvs(*d['params'], size=1)
		new_obse,_ = self.rotor.inverse_transform(new_obse, new_obs)
		new_obse = np.clip(new_obse, self.min_raw_obse, self.max_raw_obse)[0]
		if not np.all(new_obse>0):
			raise Exception(f'wrong new_obse: {new_obse}')
		return new_obse, obs
		
	def conditional_sample(self, obs):
		x = np.concatenate([np.array(self.conditional_sample_i(obs_))[None] for obs_ in obs], axis=0)
		return x[:,0], x[:,1]