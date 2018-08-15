from utils import run_extractor, EndmemberExtractor, profile_code

import argparse
import cProfile
import numpy as np

from pysptools.eea.eea import PPI

class PPIExtractor(EndmemberExtractor):
	def __init__(self, n_skewers=10000, profile=False):
		self.n_skewers = n_skewers
		self.profile = profile
		self.profile_filename = "atgp.prof"
		super(PPIExtractor, self).__init__()

	def extract_endmembers(self, hsi_3d, n_endmembers):
		hsi_2d = hsi_3d.reshape( (-1, hsi_3d.shape[2]) )
		endmembers, indicies = profile_code(self.profile,
				self.profile_filename)(PPI)(hsi_2d,
						n_endmembers,
						self.n_skewers)
		return endmembers

def main(in_filename, out_filename, n_components, profile):
	extractor = PPIExtractor(profile=profile)
	run_extractor(in_filename, out_filename, extractor, n_components)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
			description="Pixel Purity Index (PPI)",
			formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("in_filename", type=str, metavar="infile")
	parser.add_argument("out_filename", type=str, metavar="outfile")
	parser.add_argument("-n", "--n-components", type=int, default=2,
			metavar="int", help="number of coordinates for the manifold")
	parser.add_argument("--profile",
			action="store_true",
			help="profile program execution")
	
	args = vars(parser.parse_args())
	main(**args)
