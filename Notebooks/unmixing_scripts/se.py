from utils import run_extractor, AbundanceMapsExtractor, profile_code

import argparse
import cProfile
import numpy as np
from sklearn import manifold

class SEExtractor(AbundanceMapsExtractor):
	def __init__(self, args, profile):
		self.profile = profile
		self.profile_filename = "se.prof"
		self.model = manifold.SpectralEmbedding(**args)
		super(SEExtractor, self).__init__()

	def extract_abundance_maps(self, hsi_3d, n_endmembers):
		hsi_2d = hsi_3d.reshape( (-1, hsi_3d.shape[2]) )
		abundance_maps = profile_code(
				self.profile, self.profile_filename)(
						self.model.fit_transform)(hsi_2d)
		abundance_maps = np.moveaxis(abundance_maps, 1, 0)
		abundance_maps = np.reshape(abundance_maps,
				(abundance_maps.shape[0], hsi_3d.shape[0], hsi_3d.shape[1]))
		return abundance_maps

def main(in_filename, out_filename, se_args, profile):
	extractor = SEExtractor(se_args, profile)
	run_extractor(in_filename,
			out_filename, extractor, se_args["n_components"])

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Spectral Embedding", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--se-n-components",
			type=int, default=2, metavar="int",
			help="number of coordinates for the manifold")
	parser.add_argument("--se-affinity",
			type=str, default="nearest_neighbors", metavar="str",
			choices=["nearest_neighbors", "rbf", "precomputed"],
			help="how to construct the affinity matrix - choices {0}".format(["nearest_neighbors", "rbf", "precomputed"]))
	parser.add_argument("--se-gamma",
			type=float, default=None, metavar="float",
			help="kernel coefficient for rbf kernel")
	parser.add_argument("--se-random-state",
			type=int, default=None, metavar="int",
			help="seed used by the random number generator")
	parser.add_argument("--se-eigen-solver",
			type=str, default=None, metavar="str",
			choices=[None, "arpack", "lobpcg", "amg"],
			help="choices: {0}".format([None, "arpack", "lobpcg", "amg"]))
	parser.add_argument("--se-n-neighbors",
			type=int, default=None, metavar="int",
			help="number of inearest neighbors for nearest_neighbors graph building")
	parser.add_argument("--se-n-jobs",
			type=int, default=1, metavar="int",
			help="number of parallel jobs; if -1, then the number of jobs is set to the number of cores")

	parser.add_argument("in_filename", type=str, metavar="infile")
	parser.add_argument("out_filename", type=str, metavar="outfile")
	
	parser.add_argument("--profile",
			action="store_true",
			help="profile program execution")
	
	arguments = vars(parser.parse_args())
	
	args = {}
	for key,value in arguments.iteritems():
		if key.startswith("se_"):
			args.setdefault("se_args", {})[key[len("se_"):]] = value
		else:
			args[key] = value
	
	main(**args)
