import numpy as np
import lchandler._C as _Clchandler

###################################################################################################################################################

ERROR_SCALE = 1e3

EPS = 1e-5
N_TRACE_SAMPLES = 480
MAX_FIT_ERROR = 1e5
N_TUNE = 2000
THIN_BY = 20 # 10 12 # drastically affects computation time. higher, the best
SYNTH_SAMPLES_PER_CURVE = 32 # 8 16 32
CURVE_FIT_FTOL = .01
PRE_TMAX_OFFSET = 10 # 0 1 5 10 20

### OPT
REC_LOSS_EPS = 1
REC_LOSS_K = 10

### THRESHOLDS
MIN_POINTS_LIGHTCURVE_SURVEY_EXPORT = _Clchandler.MIN_POINTS_LIGHTCURVE_SURVEY_EXPORT
MIN_POINTS_LIGHTCURVE_DEFINITION = _Clchandler.MIN_POINTS_LIGHTCURVE_DEFINITION
MIN_POINTS_LIGHTCURVE_FOR_SPMFIT = 4
MIN_DUR_LIGHTCURVE_FOR_SPMFIT = 12 # 5, 10, 15, 20
MIN_SNR = -np.inf

### FILE TYPES
EXT_RAW_LIGHTCURVE = _Clchandler.EXT_RAW_LIGHTCURVE # no split, as RAW ZTF/FSNes
EXT_SPLIT_LIGHTCURVE = _Clchandler.EXT_SPLIT_LIGHTCURVE # with proper train/vali split, vali is balanced in classes
EXT_PARAMETRI_CLIGHTCURVE = 'plcd' # with sigma clipping and fitted parametric model
EXT_FATS_LIGHTCURVE = 'flcd' # with sigma clipping and FATS

### SYNTHETIC
OBSE_STD_SCALE = 1/2
MIN_CADENCE_DAYS = 2
HOURS_NOISE_AMP = 6 # to generate grid, 6 not 6/24

### DICTS
COLOR_DICT = _Clchandler.COLOR_DICT