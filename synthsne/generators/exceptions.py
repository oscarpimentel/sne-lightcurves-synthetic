from __future__ import print_function
from __future__ import division
from . import C_

###################################################################################################################################################

class TooShortCurveError(Exception):
	def __init__(self):
		pass

class TooFaintCurveError(Exception):
	def __init__(self):
		pass
		
class TraceError(Exception):
	def __init__(self):
		pass

class SyntheticCurveTimeoutError(Exception):
	def __init__(self):
		pass

class CurveFitError(Exception):
	def __init__(self):
		pass

class PYMCError(Exception):
	def __init__(self):
		pass

class InterpError(Exception):
	def __init__(self):
		pass
