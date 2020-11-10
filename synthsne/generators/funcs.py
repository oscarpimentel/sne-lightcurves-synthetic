from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np

###################################################################################################################################################

def sgm(x, x0, s):
	return 1/(1 + np.exp(-s*(x-x0)))

def syn_sne_func(t, A, t0, gamma, f, trise, tfall):
	s = 1/3
	#s = 1
	#s = 10

	g = sgm(t, gamma+t0, s)
	early = 1.0*(A*(1 - (f*(t-t0)/gamma))   /   (1 + np.exp(-(t-t0)/trise)))
	late = 1.0*(A*(1-f)*np.exp(-(t-(gamma+t0))/tfall)   /   (1 + np.exp(-(t-t0)/trise)))
	flux = (1-g)*early + g*late
	return flux

def inverse_syn_sne_func(t, A, t0, gamma, f, trise, tfall):
	return -syn_sne_func(t, A, t0, gamma, f, trise, tfall)

def syn_sne_sfunc(t, A, t0, gamma, f, trise, tfall, s):
	g = sgm(t, gamma+t0, s)
	early = 1.0*(A*(1 - (f*(t-t0)/gamma))   /   (1 + np.exp(-(t-t0)/trise)))
	late = 1.0*(A*(1-f)*np.exp(-(t-(gamma+t0))/tfall)   /   (1 + np.exp(-(t-t0)/trise)))
	flux = (1-g)*early + g*late
	return flux

def inverse_syn_sne_sfunc(t, A, t0, gamma, f, trise, tfall, s):
	return -syn_sne_sfunc(t, A, t0, gamma, f, trise, tfall, s)
