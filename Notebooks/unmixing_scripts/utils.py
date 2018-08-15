import matplotlib
matplotlib.use("Agg")

import time
from datetime import timedelta

import numpy as np

from pysptools.abundance_maps.amaps import FCLS
from pysptools.abundance_maps.amaps import NNLS

import cProfile

def timing_function(msg="Elapsed Time:"):
	def real_timing_function(function):
		def wrapper(*args, **kwargs):
			start_time = time.time()
			res = function(*args, **kwargs)
			elapsed = time.time() - start_time
			print msg, timedelta(seconds=elapsed)
			return res
		return wrapper
	return real_timing_function

def profile_code(run_profiler, out_filename, profiler=cProfile.Profile()):
	def run_profile(function):
		def wrapper(*args, **kwargs):
			try:
				profiler.enable()
				res = function(*args, **kwargs)
				profiler.disable()
				return res
			finally:
				profiler.print_stats()
				profiler.dump_stats(out_filename)
		return wrapper

	def skip_profile(function):
		def wrapper(*args, **kwargs):
			res = function(*args, **kwargs)
			return res
		return wrapper

	if run_profiler:
		return run_profile
	else:
		return skip_profile

class ComponentExtractor(object):
	ALGORITHM = None

	def __init__(self):
		pass

	@staticmethod
	def estimate_endmembers(hsi_3d, abundance_maps):
		hsi_2d = np.reshape(hsi_3d, (-1, hsi_3d.shape[2]) )
		abundance_maps = np.moveaxis(abundance_maps, 0, 2)
		abundance_maps = np.reshape(abundance_maps,
				(-1, abundance_maps.shape[2]))
		endmembers = NNLS(hsi_2d.transpose(), abundance_maps.transpose())
		endmembers = endmembers.transpose()
		return endmembers

	@staticmethod
	@timing_function(msg="abundance maps estimation time:")
	def estimate_abundance_maps(hsi_3d, endmembers):
		hsi_2d = np.reshape(hsi_3d, (-1, hsi_3d.shape[2]) )
		abundance_maps = NNLS(hsi_2d, endmembers)
		abundance_maps = np.moveaxis(abundance_maps, 1, 0)
		abundance_maps = np.reshape(abundance_maps,
				(abundance_maps.shape[0], hsi_3d.shape[0], hsi_3d.shape[1]))
		return abundance_maps

class EndmemberExtractor(ComponentExtractor):
	def __init__(self):
		super(EndmemberExtractor, self).__init__()
	
	@timing_function
	def extract_endmembers(self, hsi_3d, n_endmembers):
		raise NotImplementedError()

	def get_components(self, hsi_3d, n_components):
		endmembers = self.extract_endmembers(hsi_3d, n_components)
		abundance_maps = self.estimate_abundance_maps(hsi_3d, endmembers)
		return endmembers, abundance_maps

class AbundanceMapsExtractor(ComponentExtractor):
	def __init__(self):
		super(AbundanceMapsExtractor, self).__init__()
	
	@timing_function
	def extract_abundance_maps(self, hsi_3d, n_abundance_maps):
		raise NotImplementedError()

	def get_components(self, hsi_3d, n_components):
		abundance_maps = self.extract_abundance_maps(hsi_3d, n_components)
		endmembers = self.estimate_endmembers(hsi_3d, abundance_maps)
		return endmembers, abundance_maps

def run_extractor(in_filename, out_filename, extractor, n_components):
	with np.load(in_filename) as npzfile:
		data = npzfile["data"]
		endmembers, abundance_maps = \
				extractor.get_components(data, n_components)
	print "endmembers", endmembers.shape
	print "abundance_maps", abundance_maps.shape
	np.savez(out_filename,
		abundance_maps=abundance_maps,
		endmembers=endmembers)
	
