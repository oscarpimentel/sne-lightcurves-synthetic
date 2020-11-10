from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np

###################################################################################################################################################

def get_random_mean(a, b, r):
	assert a<=b
	assert r>=0 and r<=1
	mid = a+(b-a)/2
	return np.random.uniform(mid*(1-r), mid*(1+r))

def extract_arrays(lcobjb):
	return lcobjb.days, lcobjb.obs, lcobjb.obse