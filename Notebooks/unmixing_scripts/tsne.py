from utils import run_extractor, AbundanceMapsExtractor, profile_code

import argparse
import cProfile
import numpy as np
from sklearn import manifold

class TSNEExtractor(AbundanceMapsExtractor):
	def __init__(self, args, profile):
		self.profile = profile
		self.profile_filename = "tsne.prof"
		self.model = manifold.TSNE(**args)
		super(TSNEExtractor, self).__init__()

	def extract_abundance_maps(self, hsi_3d, n_endmembers):
		hsi_2d = hsi_3d.reshape( (-1, hsi_3d.shape[2]) )
		abundance_maps = profile_code(
				self.profile, self.profile_filename)(
						self.model.fit_transform)(hsi_2d)
		abundance_maps = np.moveaxis(abundance_maps, 1, 0)
		abundance_maps = np.reshape(abundance_maps,
				(abundance_maps.shape[0], hsi_3d.shape[0], hsi_3d.shape[1]))
		return abundance_maps

def main(in_filename, out_filename, tsne_args, profile):
	extractor = TSNEExtractor(tsne_args, profile)
	run_extractor(in_filename,
			out_filename, extractor, tsne_args["n_components"])

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
			description="Locally Linear Embeding",
			formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--tsne-n-components",
			type=int, default=2, metavar="int",
			help="number of coordinates for the manifold")
	parser.add_argument("--tsne-perplexity",
			type=float, default=30.0, metavar="float",
			help="perplexity of how well a probability distribution or probability model predicts a sample")
	parser.add_argument("--tsne-early-exaggeration",
			type=float, default=12.0, metavar="float",
			help="space between natural clusters in the embedded space")
	parser.add_argument("--tsne-learning-rate",
			type=float, default=200.0, metavar="float",
			help="space between natural clusters in the embedded space")
	parser.add_argument("--tsne-n-iter",
			type=int, default=1000, metavar="int",
			help="maximum number of iterations for the optimization; should be at least 250")
	parser.add_argument("--tsne-n-iter-without-progress",
			type=int, default=300, metavar="int",
			help="maximum number of iterations without progress before we abort the optimization, used after 250 initial iterations with early exaggeration")
	parser.add_argument("--tsne-min-grad-norm",
			type=float, default=1e-7, metavar="float",
			help="if the gradient norm is below this threshold, the optimization will be stopped")
	parser.add_argument("--tsne-metric",
			type=str, default="euclidean", metavar="str",
			help="metric to use when calculating distance between instances in a feature array")
	parser.add_argument("--tsne-init",
			type=str, default="random", metavar="str",
			choices=["random", "pca"],
			help="initialization of embedding - choices: {0}".format(["random", "pca"]))
	parser.add_argument("--tsne-verbose",
			type=int, default=0, metavar="int",
			help="level of verbosity")
	parser.add_argument("--tsne-random-state",
			type=int, default=None, metavar="int",
			help="seed used by the random number generator")
	parser.add_argument("--tsne-method",
			type=str, default="barnes_hut", metavar="str",
			choices=["barnes_hut", "exact"],
			help="gradient calculation algorithm - choices: {0}".format(["barnes_hut", "exact"]))
	parser.add_argument("--tsne-angle", 
			type=float, default=0.5, metavar="float",
			help="trade-off between speed and accuracy for Barnes-Hut T-SNE")

	parser.add_argument("in_filename", type=str, metavar="infile")
	parser.add_argument("out_filename", type=str, metavar="outfile")
	
	parser.add_argument("--profile",
			action="store_true",
			help="profile program execution")
	
	arguments = vars(parser.parse_args())
	
	args = {}
	for key,value in arguments.iteritems():
		if key.startswith("tsne_"):
			args.setdefault("tsne_args", {})[key[len("tsne_"):]] = value
		else:
			args[key] = value
	
	main(**args)
