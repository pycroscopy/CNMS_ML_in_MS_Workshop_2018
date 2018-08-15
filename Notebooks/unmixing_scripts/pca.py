from utils import run_extractor, AbundanceMapsExtractor, profile_code

import argparse
import cProfile
import numpy as np
from sklearn import decomposition

class PCAExtractor(AbundanceMapsExtractor):
	def __init__(self, args, profile):
		self.profile = profile
		self.profile_filename = "pca.prof"
		self.model = decomposition.PCA(**args)
		super(PCAExtractor, self).__init__()

	def extract_abundance_maps(self, hsi_3d, n_endmembers):
		hsi_2d = hsi_3d.reshape( (-1, hsi_3d.shape[2]) )
		abundance_maps = profile_code(
				self.profile, self.profile_filename)(
						self.model.fit_transform)(hsi_2d)
		abundance_maps = np.moveaxis(abundance_maps, 1, 0)
		abundance_maps = np.reshape(abundance_maps,
				(abundance_maps.shape[0], hsi_3d.shape[0], hsi_3d.shape[1]))
		return abundance_maps

def main(in_filename, out_filename, pca_args, profile):
	extractor = PCAExtractor(pca_args, profile)
	run_extractor(in_filename,
			out_filename, extractor, pca_args["n_components"])

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
			description="Principal Component Analysis",
			formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--pca-n-components",
			type=int, default=2, metavar="int",
			help="number of components to keep")
	parser.add_argument("--pca-whiten",
			type=bool, default=False, metavar="bool",
			help="whiten the component vectors before returning")
	parser.add_argument("--pca-svd-solver",
			type=str, default="auto", metavar="str",
			choices=["auto", "full", "arpack", "randomized"],
			help="choices: {0}".format(["auto", "full", "arpack", "randomized"]))
	parser.add_argument("--pca-tol",
			type=float, default=0.0, metavar="float",
			help="tolerance for singular values computed when svd_solver is 'arpack'")
	parser.add_argument("--pca-random-state",
			type=int, default=None, metavar="int",
			help="seed used by the random number generator")

	parser.add_argument("in_filename", type=str, metavar="infile")
	parser.add_argument("out_filename", type=str, metavar="outfile")
	
	parser.add_argument("--profile",
			action="store_true",
			help="profile program execution")

	arguments = vars(parser.parse_args())
	
	args = {}
	for key,value in arguments.iteritems():
		if key.startswith("pca_"):
			args.setdefault("pca_args", {})[key[len("pca_"):]] = value
		else:
			args[key] = value

	main(**args)
