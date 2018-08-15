from utils import run_extractor, EndmemberExtractor, profile_code

import argparse
import cProfile
import numpy as np

from pysptools.eea import NFINDR

class NFINDRExtractor(EndmemberExtractor):
	def __init__(self, profile):
		self.profile = profile
		self.profile_filename = "nfindr.prof"
		super(NFINDRExtractor, self).__init__()

	def extract_endmembers(self, hsi_3d, n_endmembers):
		extractor = NFINDR()
		endmembers = profile_code(self.profile,
				self.profile_filename)(extractor.extract)(hsi_3d,
						n_endmembers)
		return endmembers

def main(in_filename, out_filename, n_components, profile):
	extractor = NFINDRExtractor(profile)
	run_extractor(in_filename, out_filename, extractor, n_components)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
			description="N-FINDR",
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