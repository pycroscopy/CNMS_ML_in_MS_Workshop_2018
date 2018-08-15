from utils import run_extractor, AbundanceMapsExtractor, profile_code

import argparse
import cProfile
import numpy as np
from sklearn import manifold

class MDSExtractor(AbundanceMapsExtractor):
	def __init__(self, args, profile):
		self.profile = profile
		self.profile_filename = "mds.prof"
		self.model = manifold.MDS(**args)
		super(MDSExtractor, self).__init__()

	def extract_abundance_maps(self, hsi_3d, n_endmembers):
		hsi_2d = hsi_3d.reshape( (-1, hsi_3d.shape[2]) )
		abundance_maps = profile_code(
				self.profile, self.profile_filename)(
						self.model.fit_transform)(hsi_2d)
		abundance_maps = np.moveaxis(abundance_maps, 1, 0)
		abundance_maps = np.reshape(abundance_maps,
				(abundance_maps.shape[0], hsi_3d.shape[0], hsi_3d.shape[1]))
		return abundance_maps

def main(in_filename, out_filename, mds_args, profile):
	extractor = MDSExtractor(mds_args, profile)
	run_extractor(in_filename,
			out_filename, extractor, mds_args["n_components"])

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
			description="Multi-Dimensional Scaling",
			formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--mds-n-components",
			type=int, default=2, metavar="int",
			help="number of coordinates for the manifold")
	parser.add_argument("--mds-metric",
			type=bool, default=True, metavar="bool",
			help="use metric (instead of nonmetric) MDS")
	parser.add_argument("--mds-n-init",
			type=int, default=4, metavar="int",
			help="number of times the SMACOF algorithm will be run with different initializations")
	parser.add_argument("--mds-max-iter",
			type=int, default=300, metavar="int",
			help="number of times the SMACOF algorithm will be run with different initializations")
	parser.add_argument("--mds-verbose",
			type=int, default=0, metavar="int",
			help="level of verbosity")
	parser.add_argument("--mds-eps",
			type=float, default=1e-3, metavar="float",
			help="relative tolerance with respect to stress at which to declare convergence")
	parser.add_argument("--mds-n-jobs",
			type=int, default=1, metavar="int",
			help="number of parallel jobs; if -1, then the number of jobs is set to the number of cores")
	parser.add_argument("--mds-random-state",
			type=int, default=None, metavar="int",
			help="seed used by the random number generator")
	parser.add_argument("--mds-dissimilarity",
			type=str, default="euclidean", metavar="str",
			choices=["euclidean", "precomputed"],
			help="choices: {0}".format(["euclidean", "precomputed"]))

	parser.add_argument("in_filename", type=str, metavar="infile")
	parser.add_argument("out_filename", type=str, metavar="outfile")
	
	parser.add_argument("--profile",
			action="store_true",
			help="profile program execution")
	
	arguments = vars(parser.parse_args())
	
	args = {}
	for key,value in arguments.iteritems():
		if key.startswith("mds_"):
			args.setdefault("mds_args", {})[key[len("mds_"):]] = value
		else:
			args[key] = value
	
	main(**args)
