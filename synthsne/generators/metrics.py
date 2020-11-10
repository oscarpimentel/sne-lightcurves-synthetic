from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np

###################################################################################################################################################

def mse_error_syn_sne(times, obs, obse, fun, fun_args,
	scale=C_.ERROR_SCALE,
	):
	syn_obs = fun(times, *fun_args)
	error = (syn_obs-obs)**2/(obse)
	return error.mean()*scale

def wmse_error_syn_sne(times, obs, obse, fun, fun_args,
	scale=C_.ERROR_SCALE,
	):
	syn_obs = fun(times, *fun_args)
	error = (syn_obs-obs)**2/(obse)
	return error.mean()*scale

def swmse_error_syn_sne(times, obs, obse, fun, fun_args,
	scale=C_.ERROR_SCALE,
	):
	syn_obs = fun(times, *fun_args)
	error = (syn_obs-obs)**2/(obse**2)
	return error.mean()*scale