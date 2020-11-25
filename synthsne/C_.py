import numpy as np
import lchandler.C_ as C_lchandler

###################################################################################################################################################
EPS = 1e-10
N_TRACE_SAMPLES = 350
ERROR_SCALE = 1e2

### EXPORT
N_JOBS = 6 # The number of jobs to use for the computation. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
CHUNK_SIZE = N_JOBS*1

### LENGTHS & DURATIONS
MIN_POINTS_LIGHTCURVE_SURVEY_EXPORT = C_lchandler.MIN_POINTS_LIGHTCURVE_SURVEY_EXPORT
MIN_POINTS_LIGHTCURVE_DEFINITION = C_lchandler.MIN_POINTS_LIGHTCURVE_DEFINITION
MIN_POINTS_LIGHTCURVE_TO_PMFIT = 3
MIN_DUR_LIGHTCURVE_TO_PMFIT = 18

### FILE TYPES
EXT_RAW_LIGHTCURVE = C_lchandler.EXT_RAW_LIGHTCURVE # no split, as RAW ZTF/FSNes
EXT_SPLIT_LIGHTCURVE = C_lchandler.EXT_SPLIT_LIGHTCURVE # with proper train/vali split, vali is balanced in classes
EXT_PARAMETRIC_LIGHTCURVE = 'plcd' # with sigma clipping and fitted parametric model
EXT_FATS_LIGHTCURVE = 'flcd' # with sigma clipping and FATS
EXT_SAMPLER = 'smplr'

### SYNTHETIC
OBSE_STD_SCALE = C_lchandler.PLOT_OBSE_STD_SCALE
CPDS_P = 0.015 # curve points down sampling probability
HOURS_NOISE_AMP = 16.
MIN_CADENCE_DAYS = 3.
MAX_OBS_ERROR = 1e10

### DICTS
COLOR_DICT = C_lchandler.COLOR_DICT