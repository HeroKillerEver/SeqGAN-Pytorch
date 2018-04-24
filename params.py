class GenParams(object):
	"""parameters for generator"""
	emb_dim = 32
	hidden_dim = 32
	num_layers = 1



class DisParams(object):
	"""parameters for discriminator"""
	emb_dim = 64
	filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
	num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
	dropout = 0.75
		
		