import numpy as np
import lchandler.C_ as C_lchandler

###################################################################################################################################################
EPS = 1e-5
N_TRACE_SAMPLES = 480
ERROR_SCALE = 1e2
MAX_FIT_ERROR = 1e5
N_TUNE = 2000
THIN_BY = 20 # 10 12 # drastically affects computation time. higher, the best
SYNTH_SAMPLES_PER_CURVE = 12 # 8 16 32
CURVE_FIT_FTOL = .01
PRE_TMAX_OFFSET = 15 # 0 1 5 10 20

### JOBLIB
import os
JOBLIB_BACKEND = 'loky' # loky multiprocessing threading
N_JOBS = -1 # The number of jobs to use for the computation. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
CHUNK_SIZE = os.cpu_count() if N_JOBS<0 else N_JOBS

REC_LOSS_EPS = .01 # ***

### THRESHOLDS
MIN_POINTS_LIGHTCURVE_SURVEY_EXPORT = C_lchandler.MIN_POINTS_LIGHTCURVE_SURVEY_EXPORT
MIN_POINTS_LIGHTCURVE_DEFINITION = C_lchandler.MIN_POINTS_LIGHTCURVE_DEFINITION
MIN_SNR = C_lchandler.MIN_SNR
MIN_POINTS_LIGHTCURVE_FOR_SPMFIT = 4
MIN_DUR_LIGHTCURVE_FOR_SPMFIT = 12 # 5, 10, 15, 20

### FILE TYPES
EXT_RAW_LIGHTCURVE = C_lchandler.EXT_RAW_LIGHTCURVE # no split, as RAW ZTF/FSNes
EXT_SPLIT_LIGHTCURVE = C_lchandler.EXT_SPLIT_LIGHTCURVE # with proper train/vali split, vali is balanced in classes
EXT_PARAMETRIC_LIGHTCURVE = 'plcd' # with sigma clipping and fitted parametric model
EXT_FATS_LIGHTCURVE = 'flcd' # with sigma clipping and FATS

### SYNTHETIC
OBSE_STD_SCALE = 1/2 # 2 2.5 3
HOURS_NOISE_AMP = 0
MIN_CADENCE_DAYS = 2.

### DICTS
COLOR_DICT = C_lchandler.COLOR_DICT