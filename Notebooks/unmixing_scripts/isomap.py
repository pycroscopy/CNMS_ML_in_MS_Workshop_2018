from utils import run_extractor, AbundanceMapsExtractor, profile_code

import argparse
import cProfile
import numpy as np
from sklearn import manifold

class IsomapExtractor(AbundanceMapsExtractor):
	def __init__(self, args, profile):
		self.profile = profile
		self.profile_filename = "isomap.prof"
		self.model = manifold.Isomap(**args)
		super(IsomapExtractor, self).__init__()

	def extract_abundance_maps(self, hsi_3d, n_endmembers):
		hsi_2d = hsi_3d.reshape( (-1, hsi_3d.shape[2]) )
		abundance_maps = profile_code(
				self.profile, self.profile_filename)(
						self.model.fit_transform)(hsi_2d)
		abundance_maps = np.moveaxis(abundance_maps, 1, 0)
		abundance_maps = np.reshape(abundance_maps,
				(abundance_maps.shape[0], hsi_3d.shape[0], hsi_3d.shape[1]))
		return abundance_maps

def main(in_filename, out_filename, isomap_args, profile):
	extractor = IsomapExtractor(isomap_args, profile)
	run_extractor(in_filename,
			out_filename, extractor, isomap_args["n_components"])

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Isomap",
			formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--isomap-n-neighbors",
			type=int, default=5, metavar="int",
			help="number of neighbors to consider for each point")
	parser.add_argument("--isomap-n-components",
			type=int, default=2, metavar="int",
			help="number of coordinates for the manifold")
	parser.add_argument("--isomap-eigen-solver",
			type=str, default="auto", metavar="str",
			choices=["auto", "arpack", "delse"],
			help="choices: {0}".format(["auto", "arpack", "delse"]))
	parser.add_argument("--isomap-tol",
			type=float, default=0.0, metavar="float",
			help="convergence tolerance passed to arpack or lobpcg; not used if eigen_solver == 'dense'")
	parser.add_argument("--isomap-max-iter",
			type=int, default=None, metavar="int",
			help="maximum number of iterations for the arpack solver; not used if eigen_solver == 'dense'")
	parser.add_argument("--isomap-path-method",
			type=str, default="auto", metavar="str",
			choices=["auto", "FW", "D"],
			help="method to use in finding shortest path - choices: {0}".format(["auto", "FW", "D"]))
	parser.add_argument("--isomap-neighbors-algorithm",
			type=str, default="auto", metavar="str",
			choices=["auto", "brute", "kd_tree", "ball_tree"],
			help="algorithm to use for nearest neighbors search - choices {0}".format(["auto", "brute", "kd_tree", "ball_tree"]))
	parser.add_argument("--isomap-n-jobs",
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
		if key.startswith("isomap_"):
			args.setdefault("isomap_args", {})[key[len("isomap_"):]] = value
		else:
			args[key] = value
	
	main(**args)
