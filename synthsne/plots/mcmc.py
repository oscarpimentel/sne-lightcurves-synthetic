from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc3 as pm

###################################################################################################################################################

def plot_mcmc_trace(mcmc_trace_bdict, b):
	mcmc_trace = mcmc_trace_bdict[b]
	az.plot_trace(mcmc_trace)
	#pm.traceplot(mcmc_trace)
	#pm.autocorrplot(mcmc_trace)